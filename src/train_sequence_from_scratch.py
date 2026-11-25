"""Leakage-controlled account-level multi-stream sequence model."""
from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw/SA_SAML_D.csv"
EXPERIMENT=os.environ.get("AML_SEQUENCE_EXPERIMENT","sequence")
DATA=ROOT/"data"/os.environ.get("AML_SEQUENCE_DATA", "sequence")
RESULTS=ROOT/"results"/EXPERIMENT
MODEL_DIR=ROOT/"models"/("sequence_gru" if EXPERIMENT == "sequence" else EXPERIMENT)
SEQ_LEN,PAD,SEED=100,-1e6,42; NEGATIVE_ACCOUNT_RATIO=3
np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
LEAKAGE={"Is_laundering","Account_is_laundering","Laundering_type","AML_scenario_category","AML_scenario_code","AML_predicate_offence","AML_stage","AML_risk_sector","AML_scenario_source_keys","Scenario_reference_key","Scenario_detail","Scenario_label","Suspicion_indicator","Behaviour_component","Typology_stage","Legitimate_behaviour","Normal_behavior_category","Normal_behavior_source_keys","Opened_for_laundering_flag","Scenario_forced_coverage_flag","Account_laundering_probability","Transaction_purpose_category","First_observed_transaction_date","Opened_during_observation_flag","Active_observation_days","Account_activity_multiplier","Expected_transaction_count","Generated_transaction_count","KYC_strength","Beneficial_ownership_opacity"}
if os.environ.get("AML_INCLUDE_COUNTERPARTY", "1") != "1":
    LEAKAGE.add("Counterparty_type")
IDENTIFIERS={"Transaction_id","Account_id","Sender_account","Receiver_account","Counterparty_account"}

def _num(d,c): return pd.to_numeric(d[c] if c in d else 0,errors="coerce").fillna(0)

def engineer(df):
    d=df.copy(); d["event_time"]=pd.to_datetime(d["Date"].astype(str)+" "+d["Time"].astype(str),errors="coerce"); d["Account_open_date"]=pd.to_datetime(d["Account_open_date"],errors="coerce"); d=d.dropna(subset=["event_time","Account_id"]).sort_values(["Account_id","event_time","Transaction_id"],kind="stable").reset_index(drop=True); g=d.groupby("Account_id",sort=False); inflow=d.Direction.astype(str).str.lower().eq("inflow"); amount=_num(d,"Amount")
    d["role_inflow"]=inflow.astype("int8"); d["role_outflow"]=(~inflow).astype("int8"); d["hour"]=d.event_time.dt.hour; d["weekday"]=d.event_time.dt.dayofweek; d["month"]=d.event_time.dt.month
    d["hour_sin"]=np.sin(2*np.pi*d.hour/24); d["hour_cos"]=np.cos(2*np.pi*d.hour/24); d["weekday_sin"]=np.sin(2*np.pi*d.weekday/7); d["weekday_cos"]=np.cos(2*np.pi*d.weekday/7); d["month_sin"]=np.sin(2*np.pi*d.month/12); d["month_cos"]=np.cos(2*np.pi*d.month/12); d["account_age_at_event_days"]=(d.event_time-d.Account_open_date).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
    d["amount_log"]=np.log1p(amount.clip(lower=0)); d["signed_amount"]=np.where(inflow,amount,-amount); d["seconds_since_previous"]=g.event_time.diff().dt.total_seconds().fillna(0).clip(lower=0); d["previous_amount"]=g.Amount.shift(1).fillna(0); d["amount_change"]=amount-_num(d,"previous_amount")
    d["rolling_amount_mean"]=g.Amount.transform(lambda x:x.shift(1).rolling(5,min_periods=1).mean()).fillna(0); d["rolling_amount_std"]=g.Amount.transform(lambda x:x.shift(1).rolling(5,min_periods=1).std()).fillna(0); d["amount_to_rolling_mean"]=(amount/d.rolling_amount_mean.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0); d["txn_index"]=g.cumcount(); d["account_txn_count"]=g.Amount.transform("size"); d["currency_mismatch"]=(d.Payment_currency.astype(str)!=d.Received_currency.astype(str)).astype("int8")
    d["inflow_amount"]=np.where(inflow,amount,0); d["outflow_amount"]=np.where(~inflow,amount,0); d["cumulative_inflow"]=g.inflow_amount.cumsum(); d["cumulative_outflow"]=g.outflow_amount.cumsum(); d["cumulative_net_flow"]=d.cumulative_inflow-d.cumulative_outflow; d["cumulative_turnover"]=d.cumulative_inflow+d.cumulative_outflow; d["inflow_count"]=g.role_inflow.cumsum(); d["outflow_count"]=g.role_outflow.cumsum()
    d["previous_inflow_time"]=d.event_time.where(inflow).groupby(d.Account_id).ffill().groupby(d.Account_id).shift(1); d["previous_outflow_time"]=d.event_time.where(~inflow).groupby(d.Account_id).ffill().groupby(d.Account_id).shift(1); d["seconds_since_previous_inflow"]=(d.event_time-d.previous_inflow_time).dt.total_seconds().fillna(0).clip(lower=0); d["seconds_since_previous_outflow"]=(d.event_time-d.previous_outflow_time).dt.total_seconds().fillna(0).clip(lower=0); d["rapid_after_inflow_24h"]=((~inflow)&d.seconds_since_previous_inflow.between(0,86400)).astype("int8"); d["rapid_after_inflow_7d"]=((~inflow)&d.seconds_since_previous_inflow.between(0,604800)).astype("int8"); d["round_amount_proxy"]=((amount%1000).abs()<1e-6).astype("int8"); d["amount_to_cumulative_inflow"]=(amount/d.cumulative_inflow.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0); d["amount_to_cumulative_turnover"]=(amount/d.cumulative_turnover.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    d["cumulative_cross_border"]=g.Cross_border_transaction.cumsum(); d["cumulative_sadc"]=g.SADC_corridor_transaction.cumsum(); d["cumulative_virtual_asset"]=g.Virtual_asset_transaction_indicator.cumsum(); d["cumulative_procurement"]=g.Procurement_payment_indicator.cumsum()
    for c in ["Cross_border_transaction","SADC_corridor_transaction","Virtual_asset_transaction_indicator","Procurement_payment_indicator","Social_grant_indicator","Credit_card_indicator","Targeted_financial_sanctions_match","Adverse_media_flag","Public_sector_counterparty","Public_sector_exposure","Procurement_vendor","Intermediary_account_flag","Virtual_asset_exposure","Is_PEP"]:
        if c in d: d[f"account_{c}_sum"]=g[c].transform("sum"); d[f"account_{c}_max"]=g[c].transform("max")
    d["account_inflow_sum"]=g.inflow_amount.transform("sum"); d["account_outflow_sum"]=g.outflow_amount.transform("sum"); d["account_inflow_mean"]=g.inflow_amount.transform("mean"); d["account_outflow_mean"]=g.outflow_amount.transform("mean"); d["account_inflow_outflow_ratio"]=(d.account_inflow_sum/d.account_outflow_sum.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0); d["account_amount_sum"]=g.Amount.transform("sum"); d["account_amount_mean"]=g.Amount.transform("mean"); d["account_amount_std"]=g.Amount.transform("std").fillna(0); d["account_amount_max"]=g.Amount.transform("max"); d["account_unique_counterparties"]=g.Counterparty_account.transform("nunique"); d["account_unique_payment_types"]=g.Payment_type.transform("nunique"); d["account_unique_currencies"]=g.Payment_currency.transform("nunique"); d["account_elapsed_days"]=(d.event_time-g.event_time.transform("min")).dt.total_seconds().div(86400).fillna(0)
    # Labels are optional for inference on unseen data; training data carry
    # Is_laundering, while prediction files may not.
    d["account_target"] = g.Is_laundering.transform("max").astype("int8") if "Is_laundering" in d else 0
    return d

def split_accounts(df):
    a=df.groupby("Account_id",sort=False).agg(account_target=("account_target","max"),transaction_count=("Transaction_id","size")).reset_index(); dev,test=train_test_split(a,test_size=.2,stratify=a.account_target,random_state=SEED); train,val=train_test_split(dev,test_size=.2,stratify=dev.account_target,random_state=SEED); return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)

def fit_preprocessor(df,train_accounts):
    mask=df.Account_id.isin(set(train_accounts.Account_id)); excluded=LEAKAGE|IDENTIFIERS|{"Date","Time","event_time","Account_open_date","account_target","previous_inflow_time","previous_outflow_time"}; candidates=[c for c in df.columns if c not in excluded and not c.startswith("Unnamed")]; numeric=[c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]; categorical=[c for c in candidates if c not in numeric and df.loc[mask,c].nunique(dropna=False)<=100]; base=df[numeric].replace([np.inf,-np.inf],np.nan).fillna(0); scaler=RobustScaler().fit(base.loc[mask]); parts=[scaler.transform(base).astype("float32")]; low=[c for c in categorical if df.loc[mask,c].nunique(dropna=False)<=20]; high=[c for c in categorical if c not in low]
    if low: encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype="float32"); encoder.fit(df.loc[mask,low].fillna("__MISSING__").astype(str)); parts.append(encoder.transform(df[low].fillna("__MISSING__").astype(str)).astype("float32")); names=list(encoder.get_feature_names_out(low))
    else: encoder=None; names=[]
    maps={}; freq_names=[]
    for c in high: mp=df.loc[mask,c].fillna("__MISSING__").astype(str).value_counts(normalize=True).to_dict(); maps[c]=mp; parts.append(df[c].fillna("__MISSING__").astype(str).map(mp).fillna(0).to_numpy("float32")[:,None]); freq_names.append("frequency__"+c)
    X=np.column_stack(parts).astype("float32"); names=numeric+names+freq_names; joblib.dump({"numeric":numeric,"categorical":categorical,"low_cardinality":low,"high_cardinality":high,"encoder":encoder,"frequency_maps":maps,"scaler":scaler,"feature_names":names,"padding_value":PAD,"sequence_key":"Account_id","sequence_views":["all","inbound","outbound"],"counterparty_type_included":"Counterparty_type" in categorical},MODEL_DIR/"preprocessing.joblib"); return X,names

def make_sequence_stream(df,X,accounts,view="all"):
    """Create one account stream at a time to keep memory bounded."""
    groups=df.groupby("Account_id",sort=False).indices; roles=df.role_inflow.to_numpy(); out=np.full((len(accounts),SEQ_LEN,X.shape[1]),PAD,dtype="float32")
    for i,account in enumerate(accounts.Account_id):
        rows=np.asarray(groups[account],dtype="int64")
        if view=="inbound": rows=rows[roles[rows]==1]
        elif view=="outbound": rows=rows[roles[rows]==0]
        rows=rows[:SEQ_LEN]; out[i,:len(rows)]=X[rows]
    return out

def make_event_labels(df,accounts):
    """Return laundering-event labels aligned with the all-events stream."""
    groups=df.groupby("Account_id",sort=False).indices; out=np.full((len(accounts),SEQ_LEN),-1,dtype="float32"); labels=df.Is_laundering.to_numpy("float32")
    for i,account in enumerate(accounts.Account_id):
        rows=np.asarray(groups[account],dtype="int64")[:SEQ_LEN]; out[i,:len(rows)]=labels[rows]
    return out

@tf.keras.utils.register_keras_serializable(package="AML")
class MaskedMaxPool(tf.keras.layers.Layer):
    def call(self,inputs,mask=None):
        if mask is None:return tf.reduce_max(inputs,axis=1)
        m=tf.cast(mask[:,:,None],inputs.dtype); return tf.reduce_max(tf.where(m>0,inputs,tf.cast(-1e9,inputs.dtype)),axis=1)
    def compute_mask(self,inputs,mask=None): return None

@tf.keras.utils.register_keras_serializable(package="AML")
class MaskedMeanPool(tf.keras.layers.Layer):
    def call(self,inputs,mask=None):
        if mask is None:return tf.reduce_mean(inputs,axis=1)
        m=tf.cast(mask[:,:,None],inputs.dtype); return tf.reduce_sum(inputs*m,axis=1)/tf.maximum(tf.reduce_sum(m,axis=1),1.)
    def compute_mask(self,inputs,mask=None): return None

@tf.keras.utils.register_keras_serializable(package="AML")
class MaskedAttentionPool(tf.keras.layers.Layer):
    """Learn event weights while excluding post-padding positions."""
    def call(self, inputs):
        hidden, scores, reference = inputs
        valid=tf.reduce_any(tf.not_equal(reference,tf.cast(PAD,reference.dtype)),axis=-1)
        scores=tf.squeeze(scores,axis=-1)
        scores=tf.where(valid,scores,tf.cast(-1e9,scores.dtype))
        weights=tf.nn.softmax(scores,axis=1)
        return tf.reduce_sum(hidden*weights[:,:,None],axis=1)
    def compute_mask(self,inputs,mask=None): return None

def _branch(inp,name,units):
    x=tf.keras.layers.Masking(mask_value=PAD,name=f"{name}_mask")(inp); x=tf.keras.layers.Dense(64,activation="swish",kernel_regularizer=tf.keras.regularizers.l2(2e-4),name=f"{name}_projection")(x); x=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units,return_sequences=True,dropout=.12),name=f"{name}_gru")(x); return tf.keras.layers.Concatenate(name=f"{name}_pool")([MaskedMaxPool(name=f"{name}_max")(x),MaskedMeanPool(name=f"{name}_mean")(x)])

def _role_pool(inp,name):
    """Cheap mask-aware role stream; the all-events branch learns ordering."""
    x=tf.keras.layers.Masking(mask_value=PAD,name=f"{name}_mask")(inp); x=tf.keras.layers.Dense(32,activation="swish",kernel_regularizer=tf.keras.regularizers.l2(2e-4),name=f"{name}_projection")(x); return tf.keras.layers.Concatenate(name=f"{name}_pool")([MaskedMaxPool(name=f"{name}_max")(x),MaskedMeanPool(name=f"{name}_mean")(x)])

def build_model(input_dim, sequence_length=SEQ_LEN, use_attention=False):
    """Build the account-level multi-stream GRU.

    ``use_attention`` adds a learned, padding-aware event pool.  The default
    remains the original 100-event GRU so existing artefacts are unchanged;
    the attention experiment can be reproduced by calling
    ``build_model(..., use_attention=True)``.
    """
    a=tf.keras.Input((sequence_length,input_dim),name="all_events"); i=tf.keras.Input((sequence_length,input_dim),name="inbound_events"); o=tf.keras.Input((sequence_length,input_dim),name="outbound_events")
    all_hidden=tf.keras.layers.Masking(mask_value=PAD,name="all_mask")(a); all_hidden=tf.keras.layers.Dense(64,activation="swish",kernel_regularizer=tf.keras.regularizers.l2(2e-4),name="all_projection")(all_hidden); all_hidden=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(48,return_sequences=True,dropout=.15),name="all_gru")(all_hidden)
    event_logits=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation="sigmoid"),name="event_probability_raw")(all_hidden); event_out=tf.keras.layers.Reshape((sequence_length,),name="event_probability")(event_logits)
    all_max=MaskedMaxPool(name="all_max")(all_hidden); all_mean=MaskedMeanPool(name="all_mean")(all_hidden)
    if use_attention:
        attention_scores=tf.keras.layers.Dense(1,name="attention_scores")(all_hidden)
        attention_context=MaskedAttentionPool(name="masked_event_attention")([all_hidden,attention_scores,a])
        all_pool=tf.keras.layers.Concatenate(name="all_pool")([attention_context,all_max,all_mean])
    else:
        all_pool=tf.keras.layers.Concatenate(name="all_pool")([all_max,all_mean])
    x=tf.keras.layers.Concatenate()([all_pool,_role_pool(i,"inbound"),_role_pool(o,"outbound")]); x=tf.keras.layers.LayerNormalization()(x); x=tf.keras.layers.Dense(96,activation="swish",kernel_regularizer=tf.keras.regularizers.l2(2e-4))(x); x=tf.keras.layers.Dropout(.25)(x); x=tf.keras.layers.Dense(48,activation="swish",kernel_regularizer=tf.keras.regularizers.l2(2e-4))(x); account_out=tf.keras.layers.Dense(1,activation="sigmoid",name="account_probability")(x)
    def account_weighted_bce(y_true, y_pred):
        # Give laundering accounts greater influence without changing the
        # validation/test population or duplicating account histories.
        bce=tf.keras.backend.binary_crossentropy(y_true,y_pred)
        return tf.reduce_mean(bce*(1.0+2.0*tf.cast(y_true,tf.float32)),axis=-1)
    def masked_event_bce(y_true, y_pred):
        # Event labels are -1 in post-padding positions; exclude those
        # positions from the auxiliary objective rather than treating them as
        # normal events.
        valid=tf.cast(tf.greater_equal(y_true,0.0),tf.float32)
        clean=tf.where(valid>0,y_true,0.0)
        bce=tf.keras.backend.binary_crossentropy(clean,y_pred)
        return tf.reduce_sum(bce*valid,axis=-1)/tf.maximum(tf.reduce_sum(valid,axis=-1),1.0)
    m=tf.keras.Model([a,i,o],[account_out,event_out]); m.compile(tf.keras.optimizers.Adam(3e-4,clipnorm=1.),loss={"account_probability":account_weighted_bce,"event_probability":masked_event_bce},loss_weights={"account_probability":1.0,"event_probability":0.25},metrics={"account_probability":[tf.keras.metrics.AUC(name="roc_auc"),tf.keras.metrics.AUC(curve="PR",name="pr_auc"),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")],"event_probability":[tf.keras.metrics.AUC(curve="PR",name="event_pr_auc")]}); return m

def choose_threshold(y,p):
    pr,re,t=precision_recall_curve(y,p); q=pd.DataFrame({"threshold":t,"precision":pr[:-1],"recall":re[:-1]}); q["f1"]=(2*q.precision*q.recall/(q.precision+q.recall).replace(0,np.nan)).fillna(0); f=q[(q.precision>=.8)&(q.recall>=.8)]; z=f.sort_values("f1",ascending=False).iloc[0] if len(f) else q.sort_values("f1",ascending=False).iloc[0]; return float(z.threshold),q,bool(len(f))

def main():
    for p in (DATA,RESULTS,MODEL_DIR):p.mkdir(parents=True,exist_ok=True)
    df=engineer(pd.read_csv(RAW,low_memory=False)); train,val,test=split_accounts(df); X,names=fit_preprocessor(df,train); input_dim=X.shape[1]; pos=train[train.account_target==1]; neg=train[train.account_target==0].sample(n=min(len(train[train.account_target==0]),NEGATIVE_ACCOUNT_RATIO*len(pos)),random_state=SEED); sampled=pd.concat([pos,neg]).sample(frac=1,random_state=SEED).reset_index(drop=True); cohorts=(("train",sampled),("validation",val),("test",test))
    for n,a in cohorts:a.to_csv(DATA/f"{n}_accounts.csv",index=False); np.save(DATA/f"y_{n}.npy",a.account_target.to_numpy("int8")); np.save(DATA/f"y_event_{n}.npy",make_event_labels(df,a))
    for n,a in cohorts:
        for view in ("all","inbound","outbound"):
            arr=make_sequence_stream(df,X,a,view); np.save(DATA/f"X_{n}{('_'+view) if view!='all' else ''}.npy",arr); del arr
    pd.DataFrame({"Account_id":df.Account_id,"Transaction_id":df.Transaction_id,"event_time":df.event_time,"role_inflow":df.role_inflow,"account_target":df.account_target}).to_csv(DATA/"sequence_index.csv",index=False)
    del X,df
    import gc; gc.collect()
    tr=[np.load(DATA/f"X_train{('_'+view) if view!='all' else ''}.npy",mmap_mode="r") for view in ("all","inbound","outbound")]; va=[np.load(DATA/f"X_validation{('_'+view) if view!='all' else ''}.npy",mmap_mode="r") for view in ("all","inbound","outbound")]; te=[np.load(DATA/f"X_test{('_'+view) if view!='all' else ''}.npy",mmap_mode="r") for view in ("all","inbound","outbound")]; ytr=np.load(DATA/"y_train.npy"); yv=np.load(DATA/"y_validation.npy"); yt=np.load(DATA/"y_test.npy"); evtr=np.load(DATA/"y_event_train.npy",mmap_mode="r"); evv=np.load(DATA/"y_event_validation.npy",mmap_mode="r"); evt=np.load(DATA/"y_event_test.npy",mmap_mode="r")
    m=build_model(input_dim); train_inputs={"all_events":tr[0],"inbound_events":tr[1],"outbound_events":tr[2]}; val_inputs={"all_events":va[0],"inbound_events":va[1],"outbound_events":va[2]}; test_inputs={"all_events":te[0],"inbound_events":te[1],"outbound_events":te[2]}; h=m.fit(train_inputs,{"account_probability":ytr,"event_probability":evtr},validation_data=(val_inputs,{"account_probability":yv,"event_probability":evv}),epochs=20,batch_size=512,shuffle=True,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_account_probability_pr_auc",mode="max",patience=6,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_account_probability_pr_auc",mode="max",patience=2,factor=.5,min_lr=1e-6)]); m.save(MODEL_DIR/"model.keras"); hist=pd.DataFrame(h.history); hist.to_csv(RESULTS/"training_history.csv",index=False)
    pv=m.predict(val_inputs,batch_size=512,verbose=0)[0].ravel(); pt=m.predict(test_inputs,batch_size=512,verbose=0)[0].ravel(); threshold,candidates,feasible=choose_threshold(yv,pv); pred=(pt>=threshold).astype(int); report=pd.DataFrame(classification_report(yt,pred,labels=[0,1],target_names=["Normal","Laundering"],output_dict=True,zero_division=0)).T; report.to_csv(RESULTS/"classification_report.csv"); tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel(); metrics={"unit":"Account_id","sequence_views":"all+inbound+outbound","auxiliary_event_task":True,"counterparty_type_included":"Counterparty_type" not in LEAKAGE,"sequence_length":SEQ_LEN,"padding":"post","padding_value":PAD,"threshold":threshold,"roc_auc":roc_auc_score(yt,pt),"average_precision":average_precision_score(yt,pt),"true_negative":int(tn),"false_positive":int(fp),"false_negative":int(fn),"true_positive":int(tp),"precision_positive":float(report.loc["Laundering","precision"]),"recall_positive":float(report.loc["Laundering","recall"]),"feasible_0.8_0.8_on_validation":feasible,"train_accounts":int(len(sampled)),"validation_accounts":int(len(val)),"test_accounts":int(len(test)),"input_features":int(input_dim)}; pd.DataFrame([metrics]).to_csv(RESULTS/"test_metrics.csv",index=False); candidates.to_csv(RESULTS/"threshold_candidates.csv",index=False); pd.DataFrame({"Account_id":test.Account_id,"y_true":yt,"probability":pt,"prediction":pred}).to_csv(RESULTS/"test_predictions.csv",index=False); (MODEL_DIR/"model_manifest.json").write_text(json.dumps({"sequence_key":"Account_id","sequence_views":["all","inbound","outbound"],"auxiliary_event_task":True,"counterparty_type_included":"Counterparty_type" not in LEAKAGE,"sequence_length":SEQ_LEN,"padding_value":PAD,"padding_position":"post","input_features":names,"leakage_excluded":sorted(LEAKAGE|IDENTIFIERS),"negative_sampling":"all positive training accounts plus up to three normal accounts per positive","class_weight":{"account_positive":2.0,"event_positive":2.0},"event_label_padding":-1,"metrics":metrics},indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2)); print(report.to_string())

if __name__=="__main__":main()
