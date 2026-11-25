"""Validation-fitted probability calibration for transaction XGBoost."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"transaction_xgb"; RESULTS=ROOT/"results"/"transaction_xgb"; OUT=RESULTS/"calibration"; FIG=OUT/"figures"; MODEL=ROOT/"models"/"transaction_xgb"; EPS=1e-6
def logit(p): p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def sig(x): return 1/(1+np.exp(-np.clip(np.asarray(x,float),-40,40)))
def beta_x(p): p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.c_[np.log(p),np.log1p(-p)]
def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); out=0
    for i in range(bins):
        m=(p>=edges[i])&((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any(): out+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    model=XGBClassifier(); model.load_model(MODEL/"model.json"); yv=np.load(DATA/"y_validation.npy").astype(int); yt=np.load(DATA/"y_test.npy").astype(int); pv=model.predict_proba(sp.load_npz(DATA/"X_validation.npz"))[:,1]; pt=model.predict_proba(sp.load_npz(DATA/"X_test.npz"))[:,1]
    lv=logit(pv); platt=LogisticRegression(solver="lbfgs").fit(lv[:,None],yv); beta=LogisticRegression(solver="lbfgs").fit(beta_x(pv),yv); temp=float(minimize_scalar(lambda t:log_loss(yv,sig(lv/t),labels=[0,1]),bounds=(.05,10),method="bounded").x)
    probs={"uncalibrated":pt,"platt":platt.predict_proba(logit(pt)[:,None])[:,1],"beta":beta.predict_proba(beta_x(pt))[:,1],"temperature":sig(logit(pt)/temp)}; rows=[]
    for name,p in probs.items(): rows.append({"method":name,"brier_score":brier_score_loss(yt,p),"log_loss":log_loss(yt,p,labels=[0,1]),"ece_10_bins":ece(yt,p)})
    pd.DataFrame(rows).to_csv(OUT/"calibration_metrics.csv",index=False); out=pd.DataFrame({"y_true":yt,"probability_uncalibrated":pt}); [out.__setitem__(f"probability_{k}",v) for k,v in probs.items() if k!="uncalibrated"]; out.to_csv(OUT/"test_calibrated_predictions.csv",index=False); joblib.dump({"platt":platt,"beta":beta,"temperature":temp,"isotonic_used":False},MODEL/"calibrators.joblib"); (MODEL/"calibration_manifest.json").write_text(json.dumps({"fit_split":"validation","evaluation_split":"test","methods":["uncalibrated","platt","beta","temperature"],"isotonic_used":False,"temperature":temp},indent=2),encoding="utf-8")
    colors={"uncalibrated":"#2878b5","platt":"#e67e22","beta":"#2ca25f","temperature":"#8e44ad"}; fig,ax=plt.subplots(figsize=(7,5)); ax.plot([0,1],[0,1],"k--",label="Perfect calibration")
    for name,p in probs.items():
        frac,mean=calibration_curve(yt,p,n_bins=10,strategy="uniform"); ax.plot(mean,frac,marker="o",label=name,color=colors[name]); [ax.annotate(f"{yy:.2f}",(xx,yy),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7,color=colors[name]) for xx,yy in zip(mean,frac)]
    ax.set(xlabel="Mean predicted probability",ylabel="Observed frequency",title="Transaction XGBoost calibration (10 bins)"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/"calibration_curves.png",dpi=300); plt.close(fig)
    for metric in ["brier_score","log_loss","ece_10_bins"]:
        vals=[r[metric] for r in rows]; fig,ax=plt.subplots(figsize=(7,4)); bars=ax.bar([r["method"] for r in rows],vals,color=[colors[r["method"]] for r in rows]); ax.set(title=f"Transaction XGBoost calibration: {metric}",ylabel=metric); ax.grid(axis="y",alpha=.2); [ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.4f}",ha="center",va="bottom",fontsize=8) for b in bars]; fig.tight_layout(); fig.savefig(FIG/f"{metric}.png",dpi=300); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
