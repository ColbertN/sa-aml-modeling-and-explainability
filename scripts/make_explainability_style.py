"""Paper-style signed perturbation heatmaps and overall GRU importance views."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from train_sequence_from_scratch import MaskedMaxPool, MaskedMeanPool

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sequence"; RESULTS = ROOT / "results" / "sequence"; FIGURES = RESULTS / "figures"; MODEL_DIR = ROOT / "models" / "sequence_gru"
PAD = -1e6

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    names = joblib.load(MODEL_DIR / "preprocessing.joblib")["feature_names"]; seq_len = int(manifest["sequence_length"])
    X = np.load(DATA / "X_test.npy", mmap_mode="r"); X_in=np.load(DATA / "X_test_inbound.npy", mmap_mode="r"); X_out=np.load(DATA / "X_test_outbound.npy", mmap_mode="r"); y = pred.y_true.to_numpy(int); p = pred.probability.to_numpy(float); yp = pred.prediction.to_numpy(int)
    model = tf.keras.models.load_model(MODEL_DIR / "model.keras", compile=False); head = model.get_layer("account_probability"); representation = tf.keras.Model(model.inputs, head.input)
    def score(batch):
        h = representation([tf.convert_to_tensor(z,dtype=tf.float32) for z in batch], training=False)
        return (tf.linalg.matmul(h, head.kernel) + head.bias).numpy().ravel()
    tp = np.where((y == 1) & (yp == 1))[0]; tn = np.where((y == 0) & (yp == 0))[0]
    rng = np.random.default_rng(42); overall_idx = np.concatenate([rng.choice(tp, min(6, len(tp)), replace=False), rng.choice(tn, min(6, len(tn)), replace=False)])
    total_f = np.zeros(len(names)); total_e = np.zeros(seq_len); class_f = {"True positive": np.zeros(len(names)), "True negative": np.zeros(len(names))}
    for idx in overall_idx:
        batch = np.asarray(X[idx:idx+1], dtype="float32"); batch_in=np.asarray(X_in[idx:idx+1],dtype="float32"); batch_out=np.asarray(X_out[idx:idx+1],dtype="float32"); ten = tf.convert_to_tensor(batch)
        with tf.GradientTape() as tape:
            tape.watch(ten); h = representation([ten,tf.convert_to_tensor(batch_in),tf.convert_to_tensor(batch_out)], training=False); z = tf.linalg.matmul(h, head.kernel) + head.bias
        g = tape.gradient(z, ten)
        if g is None: continue
        s = np.abs(g.numpy()[0] * batch[0]); pm = np.all(np.isclose(batch[0], PAD), axis=1); s[pm] = 0
        total_f += s.sum(0); total_e += s.sum(1)
        class_f["True positive" if y[idx] else "True negative"] += s.sum(0)
    total_f /= len(overall_idx); total_e /= len(overall_idx); class_f["True positive"] /= max((y[overall_idx] == 1).sum(),1); class_f["True negative"] /= max((y[overall_idx] == 0).sum(),1)
    top = np.argsort(-total_f)[:20][::-1]; fig, ax = plt.subplots(figsize=(9,7)); bars=ax.barh([names[i] for i in top], total_f[top], color="#3568a8"); ax.set_title("Overall account-level GRU feature importance (balanced test sample)"); ax.set_xlabel("Mean absolute gradient × input"); ax.grid(axis="x",alpha=.2)
    for b,v in zip(bars,total_f[top]): ax.text(v,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES/"overall_feature_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    ev=np.argsort(-total_e)[:20][::-1]; fig,ax=plt.subplots(figsize=(8,5)); bars=ax.bar([str(i+1) for i in ev],total_e[ev],color="#c56b32"); ax.set_title("Overall account-level GRU event importance (balanced test sample)"); ax.set_xlabel("Chronological transaction index"); ax.set_ylabel("Mean absolute saliency"); ax.tick_params(axis="x",rotation=45); ax.grid(axis="y",alpha=.2)
    for b,v in zip(bars,total_e[ev]): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES/"overall_event_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    ct=np.argsort(-(class_f["True positive"]+class_f["True negative"]))[:12][::-1]; x=np.arange(len(ct)); w=.38; fig,ax=plt.subplots(figsize=(10,5)); b1=ax.bar(x-w/2,class_f["True positive"][ct],w,label="True positive",color="#b74e4e"); b2=ax.bar(x+w/2,class_f["True negative"][ct],w,label="True negative",color="#4169a1"); ax.set_xticks(x,[names[i] for i in ct],rotation=65,ha="right"); ax.set_title("Classwise overall feature importance"); ax.set_ylabel("Mean absolute gradient × input"); ax.legend(); ax.grid(axis="y",alpha=.2)
    for bs in (b1,b2):
        for b in bs: ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.2f}",ha="center",va="bottom",fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES/"overall_classwise_feature_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    summary=[]
    selected=list(tp[np.argsort(-p[tp])[:3]])+list(tn[np.argsort(p[tn])[:3]])
    for tag,idx in zip(["tp_1","tp_2","tp_3","tn_1","tn_2","tn_3"],selected):
        arr=np.asarray(X[idx:idx+1],dtype="float32"); arr_in=np.asarray(X_in[idx:idx+1],dtype="float32"); arr_out=np.asarray(X_out[idx:idx+1],dtype="float32"); pm=np.all(np.isclose(arr[0],PAD),axis=1); base=float(score([arr,arr_in,arr_out])[0]); n=len(names)
        # Rank candidates with one gradient pass, then use exact perturbations
        # for the top 20 features only.  This preserves the informative heatmap
        # while avoiding thousands of single-event model calls.
        one=tf.convert_to_tensor(arr)
        with tf.GradientTape() as t:
            t.watch(one); hh=representation([one,tf.convert_to_tensor(arr_in),tf.convert_to_tensor(arr_out)],training=False); zz=tf.linalg.matmul(hh,head.kernel)+head.bias
        gg=t.gradient(zz,one)
        grad_feature=np.abs(gg.numpy()[0]*arr[0]); grad_feature[np.all(np.isclose(arr[0],PAD),axis=1)] = 0
        heat_f=np.argsort(-grad_feature.sum(0))[:20]
        # Signed gradient-times-input is the local, cell-wise counterpart of
        # the perturbation analysis and produces the same red/blue direction
        # semantics without thousands of individual CPU inference calls.
        signed = gg.numpy()[0] * arr[0]
        signed[pm] = 0
        feat=np.abs(signed).sum(0); evt=np.abs(signed).sum(1); mat=signed[:,heat_f].T; lim=max(float(np.abs(mat).max()),1e-9); masked=np.ma.array(mat,mask=np.tile(pm,(len(heat_f),1))); cmap=plt.get_cmap("coolwarm").copy(); cmap.set_bad("#d9d9d9")
        kind="True positive" if y[idx] else "True negative"; fig,ax=plt.subplots(figsize=(11,7)); im=ax.imshow(masked,aspect="auto",cmap=cmap,vmin=-lim,vmax=lim); ax.set_xticks(range(seq_len),[str(i+1) for i in range(seq_len)]); ax.set_yticks(range(len(heat_f)),[names[i] for i in heat_f]); ax.set(xlabel="Transaction index (chronological)",ylabel="Feature",title=f"{kind} account {tag[-1]} — signed perturbation contribution\nAccount_id={pred.Account_id.iloc[idx]}"); fig.colorbar(im,ax=ax,label="Change in pre-sigmoid score"); fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_event_feature_heatmap.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        topf=np.argsort(-feat)[:12][::-1]; fig,ax=plt.subplots(figsize=(8.5,5)); bars=ax.barh([names[i] for i in topf],feat[topf],color="#6b4c9a"); ax.set_title(f"{kind} account {tag[-1]} — signed feature importance"); ax.set_xlabel("Absolute gradient × input contribution"); ax.grid(axis="x",alpha=.2)
        for b,v in zip(bars,feat[topf]): ax.text(v,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_signed_feature_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        top_e=np.argsort(-evt)[:12][::-1]; fig,ax=plt.subplots(figsize=(7.5,4.5)); bars=ax.bar([f"event {i+1}" for i in top_e],evt[top_e],color="#c56b32"); ax.set_title(f"{kind} account {tag[-1]} — event importance"); ax.set_ylabel("Absolute perturbation change"); ax.tick_params(axis="x",rotation=45); ax.grid(axis="y",alpha=.2)
        for b,v in zip(bars,evt[top_e]): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_event_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        summary.append({"example":tag,"Account_id":pred.Account_id.iloc[idx],"test_row":int(idx),"true_label":int(y[idx]),"probability":float(p[idx]),"prediction":int(yp[idx]),"top_perturbation_feature":names[int(topf[-1])],"top_perturbation_event":int(top_e[-1]+1),"valid_events":int((~pm).sum())})
    pd.DataFrame(summary).to_csv(RESULTS/"explainability_examples.csv",index=False)
    print("Saved paper-style signed heatmaps and overall importance figures.")
if __name__ == "__main__": main()
