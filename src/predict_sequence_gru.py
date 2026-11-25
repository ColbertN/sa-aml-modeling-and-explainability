"""Score unseen transaction CSV data with the persisted Account_id GRU."""
from pathlib import Path
import json, sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from train_sequence_from_scratch import MaskedMaxPool, MaskedMeanPool, engineer, make_sequence_stream, SEQ_LEN

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/sequence"; MODEL_DIR=ROOT/"models/sequence_gru"

def main(input_path, output_path):
    df=engineer(pd.read_csv(input_path,low_memory=False)); prep=joblib.load(MODEL_DIR/"preprocessing.joblib"); numeric=prep["numeric"]; low=prep["low_cardinality"]; high=prep["high_cardinality"]; base=df[numeric].replace([np.inf,-np.inf],np.nan).fillna(0); xn=prep["scaler"].transform(base).astype("float32"); parts=[xn]
    if low: parts.append(prep["encoder"].transform(df[low].fillna("__MISSING__").astype(str)).astype("float32"))
    for c in high: parts.append(df[c].fillna("__MISSING__").astype(str).map(prep["frequency_maps"][c]).fillna(0).to_numpy("float32")[:,None])
    X=np.column_stack(parts).astype("float32"); accounts=df.groupby("Account_id",sort=False).agg(account_target=("account_target","max")).reset_index(); streams=[make_sequence_stream(df,X,accounts,view) for view in ("all","inbound","outbound")]; model=tf.keras.models.load_model(MODEL_DIR/"model.keras",compile=False); raw=model.predict(streams,batch_size=512,verbose=0); p=(raw[0] if isinstance(raw,(list,tuple)) else raw).ravel(); manifest=json.loads((MODEL_DIR/"model_manifest.json").read_text(encoding="utf-8")); threshold=float(manifest["metrics"]["threshold"]); out=pd.DataFrame({"Account_id":accounts.Account_id.to_numpy(),"probability_laundering":p,"prediction":(p>=threshold).astype(int)}); out.to_csv(output_path,index=False); print(f"Saved {len(out):,} account predictions to {output_path}")

if __name__=="__main__":
    if len(sys.argv)!=3: raise SystemExit("Usage: py -3.11 src/predict_sequence_gru.py input.csv output.csv")
    main(Path(sys.argv[1]),Path(sys.argv[2]))
