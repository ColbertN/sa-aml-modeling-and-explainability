"""Clean account-level event contribution figures (padding omitted)."""
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
    X = np.load(DATA / "X_test.npy", mmap_mode="r"); X_in=np.load(DATA / "X_test_inbound.npy", mmap_mode="r"); X_out=np.load(DATA / "X_test_outbound.npy", mmap_mode="r"); y = pred.y_true.to_numpy(int); yp = pred.prediction.to_numpy(int); prob = pred.probability.to_numpy(float)
    model = tf.keras.models.load_model(MODEL_DIR / "model.keras", compile=False); head = model.get_layer("account_probability"); representation = tf.keras.Model(model.inputs, head.input)
    def contribution(index):
        arr = np.asarray(X[index:index+1], dtype="float32"); arr_in=np.asarray(X_in[index:index+1],dtype="float32"); arr_out=np.asarray(X_out[index:index+1],dtype="float32"); ten = tf.convert_to_tensor(arr)
        with tf.GradientTape() as tape:
            tape.watch(ten); h = representation([ten,tf.convert_to_tensor(arr_in),tf.convert_to_tensor(arr_out)], training=False); z = tf.linalg.matmul(h, head.kernel) + head.bias
        signed = tape.gradient(z, ten).numpy()[0] * arr[0]; pad_mask = np.all(np.isclose(arr[0], PAD), axis=1); signed[pad_mask] = 0
        return signed, pad_mask
    tp = np.where((y == 1) & (yp == 1))[0]; tn = np.where((y == 0) & (yp == 0))[0]
    selected = list(tp[np.argsort(-prob[tp])[:3]]) + list(tn[np.argsort(prob[tn])[:3]])
    for tag, index in zip(["tp_1", "tp_2", "tp_3", "tn_1", "tn_2", "tn_3"], selected):
        signed, pad_mask = contribution(index); actual = np.where(~pad_mask)[0]
        if len(actual) == 0: continue
        values = signed[actual]; event_values = values.sum(axis=1); importance = np.abs(event_values); feature_values = np.abs(values).sum(axis=0); top_features = np.argsort(-feature_values)[:12]
        kind = "True positive" if y[index] else "True negative"; account = pred.Account_id.iloc[index]; n_actual = len(actual); ticks = list(range(0, n_actual, max(1, int(np.ceil(n_actual/10))))); width = max(7.5, min(14, 5.5 + .18*n_actual))
        # Timeline over actual transactions only.
        fig, ax = plt.subplots(figsize=(width,5)); bars=ax.bar(np.arange(n_actual),event_values,color=np.where(event_values>=0,"#c94c4c","#426fb0"),edgecolor="none"); ax.set_xticks(ticks,[str(i+1) for i in ticks]); ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Actual transaction index (chronological)",ylabel="Signed contribution",title=f"{kind} account - event contribution timeline\nAccount_id={account}"); ax.grid(axis="y",alpha=.2)
        for e in np.argsort(-np.abs(event_values))[:min(5,n_actual)]:
            offset = 8 if event_values[e] >= 0 else -10
            ax.annotate(f"{event_values[e]:.2f}", (e, event_values[e]), xytext=(0, offset), textcoords="offset points", ha="center", va="bottom" if event_values[e] >= 0 else "top", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_event_timeline.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        # Event-importance bars use the same local transaction indices.
        fig, ax = plt.subplots(figsize=(width,5)); bars=ax.bar(np.arange(n_actual),importance,color="#c56b32",edgecolor="none"); ax.set_xticks(ticks,[str(i+1) for i in ticks]); ax.set(xlabel="Actual transaction index (chronological)",ylabel="Absolute contribution",title=f"{kind} account - event importance\nAccount_id={account}"); ax.grid(axis="y",alpha=.2)
        for e in np.argsort(-importance)[:min(5,n_actual)]:
            ax.annotate(f"{importance[e]:.2f}", (e, importance[e]), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_event_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        # Cropped signed heatmap.
        heat=values[:,top_features].T; limit=max(float(np.abs(heat).max()),1e-9); cmap=plt.get_cmap("coolwarm").copy(); cmap.set_bad("#ffffff"); fig,ax=plt.subplots(figsize=(width,6.2)); im=ax.imshow(heat,aspect="auto",cmap=cmap,vmin=-limit,vmax=limit); ax.set_xticks(ticks,[str(i+1) for i in ticks]); ax.set_yticks(range(len(top_features)),[names[i] for i in top_features]); ax.set(xlabel="Actual transaction index (chronological)",ylabel="Feature",title=f"{kind} account - signed feature/event contributions\nAccount_id={account} (padding omitted)"); fig.colorbar(im,ax=ax,label="Signed gradient x input contribution")
        for flat in np.argsort(-np.abs(heat).ravel())[:min(10,heat.size)]: rr,cc=np.unravel_index(flat,heat.shape); ax.text(cc,rr,f"{heat[rr,cc]:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(heat[rr,cc])>.35*limit else "black")
        fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_event_feature_heatmap.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        # Cropped feature/event dot plot.
        dot=values[:,top_features].T; scale=max(float(np.abs(dot).max()),1e-9); xx,yy=np.meshgrid(np.arange(n_actual),np.arange(len(top_features))); vv=dot.ravel(); fig,ax=plt.subplots(figsize=(width,6)); sc=ax.scatter(xx.ravel(),yy.ravel(),s=30+420*np.abs(vv)/scale,c=vv,cmap="coolwarm",vmin=-scale,vmax=scale,alpha=.85,edgecolors="white",linewidths=.25); ax.set_xticks(ticks,[str(i+1) for i in ticks]); ax.set_yticks(range(len(top_features)),[names[i] for i in top_features]); ax.set(xlabel="Actual transaction index (chronological)",ylabel="Top features",title=f"{kind} account - feature/event contribution dot plot\nAccount_id={account} (padding omitted)"); fig.colorbar(sc,ax=ax,label="Signed gradient x input contribution")
        for flat in np.argsort(-np.abs(vv))[:min(10,len(vv))]: rr,cc=np.unravel_index(flat,dot.shape); ax.annotate(f"{vv[flat]:.2f}",(cc,rr),xytext=(3,3),textcoords="offset points",fontsize=7)
        fig.tight_layout(); fig.savefig(FIGURES/f"{tag}_event_feature_dotplot.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    print("Saved cropped timelines, event-importance plots, heatmaps and dot plots for six accounts.")

if __name__ == "__main__": main()
