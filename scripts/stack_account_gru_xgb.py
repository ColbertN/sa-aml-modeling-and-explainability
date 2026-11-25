"""Validation-fitted account-level stack of GRU and transaction XGBoost scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from train_sequence_from_scratch import MaskedMaxPool, MaskedMeanPool, MaskedAttentionPool, engineer
import joblib

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"; OUT=ROOT/"results/sequence_hybrid"; MODEL=ROOT/"models/sequence_hybrid"; PAD=-1e6

def account_xgb_scores(raw, account_ids):
    from train_transaction_xgb import engineer as xeng
    d=xeng(raw); ids=set(account_ids); d=d[d.Account_id.isin(ids)].copy(); prep=joblib.load(ROOT/"models/transaction_xgb/preprocessing.joblib"); use=prep["numeric"]+prep["categorical"]; X=prep["pipeline"].transform(d[use]); model=XGBClassifier(); model.load_model(ROOT/"models/transaction_xgb/model.json"); p=model.predict_proba(X)[:,1]; d["p"]=p; return d.groupby("Account_id").p.max()

def choose(y, s):
    """Select an operating point using validation data only.

    The guard-band (precision >= .82 and recall >= .80) is deliberate: it
    protects the requested 0.80/0.80 operating objective against modest
    validation-to-test drift without inspecting test labels.
    """
    pr,re,t=precision_recall_curve(y,s); q=pd.DataFrame({"threshold":t,"precision":pr[:-1],"recall":re[:-1]}); q["f1"]=2*q.precision*q.recall/(q.precision+q.recall).replace(0,np.nan); q["f1"]=q.f1.fillna(0); feasible=q[(q.precision>=.82)&(q.recall>=.80)]; row=(feasible.sort_values("f1",ascending=False).iloc[0] if len(feasible) else q.sort_values("f1",ascending=False).iloc[0]); return float(row.threshold),q,bool(len(feasible))

def main():
    OUT.mkdir(parents=True,exist_ok=True); MODEL.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(DATA/"raw/SA_SAML_D.csv",low_memory=False); gru=tf.keras.models.load_model(ROOT/"models/sequence_gru/model.keras",compile=False); xv=np.load(DATA/"sequence/X_validation.npy"); xt=np.load(DATA/"sequence/X_test.npy"); pgv=gru.predict(xv,batch_size=512,verbose=0).ravel(); pgt=gru.predict(xt,batch_size=512,verbose=0).ravel(); av=pd.read_csv(DATA/"sequence/validation_accounts.csv"); at=pd.read_csv(DATA/"sequence/test_accounts.csv"); yv=av.account_target.to_numpy(int); yt=at.account_target.to_numpy(int); xsv=account_xgb_scores(raw,av.Account_id); xst=account_xgb_scores(raw,at.Account_id); xv_score=xsv.reindex(av.Account_id).fillna(0).to_numpy(); xt_score=xst.reindex(at.Account_id).fillna(0).to_numpy()
    best=None; all_rows=[]
    for alpha in np.linspace(0,1,21):
        s=alpha*pgv+(1-alpha)*xv_score; th,q,feas=choose(yv,s); row=q[q.threshold==th].iloc[0]; all_rows.append({"gru_weight":alpha,"xgb_weight":1-alpha,"threshold":th,"precision":row.precision,"recall":row.recall,"f1":row.f1,"meets_0.8_0.8":feas}); key=(feas,row.f1,min(row.precision,row.recall));
        if best is None or key>(best[0],best[1],best[2]): best=(feas,min(row.precision,row.recall),row.f1,alpha,th)
    _,_,_,alpha,threshold=best; score_v=alpha*pgv+(1-alpha)*xv_score; score_t=alpha*pgt+(1-alpha)*xt_score; pred=(score_t>=threshold).astype(int); report=pd.DataFrame(classification_report(yt,pred,labels=[0,1],target_names=["Normal","Laundering"],output_dict=True,zero_division=0)).T; tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel(); metrics={"unit":"Account_id hybrid","gru_weight":alpha,"xgb_weight":1-alpha,"threshold":threshold,"roc_auc":roc_auc_score(yt,score_t),"average_precision":average_precision_score(yt,score_t),"true_negative":int(tn),"false_positive":int(fp),"false_negative":int(fn),"true_positive":int(tp),"precision_positive":float(report.loc["Laundering","precision"]),"recall_positive":float(report.loc["Laundering","recall"]),"validation_meets_0.8_0.8":bool([r for r in all_rows if r["gru_weight"]==alpha][0]["meets_0.8_0.8"])}; pd.DataFrame([metrics]).to_csv(OUT/"test_metrics.csv",index=False); report.to_csv(OUT/"classification_report.csv"); pd.DataFrame({"Account_id":at.Account_id,"y_true":yt,"gru_probability":pgt,"xgb_max_probability":xt_score,"hybrid_probability":score_t,"prediction":pred}).to_csv(OUT/"test_predictions.csv",index=False); pd.DataFrame({"Account_id":av.Account_id,"y_true":yv,"gru_probability":pgv,"xgb_max_probability":xv_score,"hybrid_probability":score_v}).to_csv(OUT/"validation_predictions.csv",index=False); pd.DataFrame(all_rows).to_csv(OUT/"validation_blend_search.csv",index=False); (MODEL/"hybrid_manifest.json").write_text(json.dumps({"components":["models/sequence_gru/model.keras","models/transaction_xgb/model.json"],"score":"alpha*GRU_account_probability + (1-alpha)*max_transaction_XGBoost_probability","fit_split":"validation","evaluation_split":"test","selection_policy":"validation precision >= 0.82 and recall >= 0.80; maximise F1","alpha_gru":alpha,"alpha_xgb":1-alpha,"threshold":threshold,"metrics":metrics},indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2)); print(report.to_string())

if __name__=="__main__": main()
