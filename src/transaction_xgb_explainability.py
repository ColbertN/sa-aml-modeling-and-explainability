"""Global and local SHAP plus perturbation explanations for transaction XGBoost."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transaction_xgb"; MODEL_DIR = ROOT / "models" / "transaction_xgb"
RESULTS = ROOT / "results" / "transaction_xgb" / "explainability"; FIG = RESULTS / "figures"


def clean_name(name):
    return name.replace("num__", "").replace("cat__", "")


def bar(values, names, title, path, color="#2b6cb0", xlabel="Mean absolute contribution"):
    order = np.argsort(values)[-min(20, len(values)):][::-1]
    vals = np.asarray(values)[order]; labels = [clean_name(names[i]) for i in order]
    fig, ax = plt.subplots(figsize=(9, max(4.5, .32 * len(labels))))
    y = np.arange(len(labels)); ax.barh(y, vals[::-1], color=color); ax.set_yticks(y, labels[::-1]); ax.set_xlabel(xlabel); ax.set_title(title, weight="bold"); ax.grid(axis="x", alpha=.2)
    for yy, vv in zip(y, vals[::-1]): ax.text(vv, yy, f" {vv:.4f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "model_manifest.json", encoding="utf-8") as f: manifest = json.load(f)
    names = manifest["feature_names"]
    model = XGBClassifier(); model.load_model(MODEL_DIR / "model.json")
    X = sp.load_npz(DATA / "X_test.npz").tocsr(); pred = pd.read_csv(ROOT / "results" / "transaction_xgb" / "test_predictions.csv")
    y = pred.y_true.to_numpy(int); p = pred.probability.to_numpy(float); threshold = float(pd.read_csv(ROOT / "results" / "transaction_xgb" / "test_metrics.csv").iloc[0].threshold); yh = (p >= threshold).astype(int)
    rng = np.random.default_rng(42); n = min(1800, X.shape[0]); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0); take_pos = rng.choice(pos, min(n // 2, len(pos)), replace=False); take_neg = rng.choice(neg, n - len(take_pos), replace=False); idx = np.concatenate([take_pos, take_neg]); rng.shuffle(idx)
    dense = X[idx].toarray().astype(np.float32)
    explainer = shap.TreeExplainer(model); sv = explainer.shap_values(dense); sv = sv[0] if isinstance(sv, list) else np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0); imp = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False); imp.to_csv(RESULTS / "global_shap_importance.csv", index=False)
    bar(mean_abs, names, "Transaction XGBoost global SHAP importance", FIG / "global_shap_bar.png", xlabel="Mean absolute SHAP value")
    plt.figure(figsize=(10, 7)); shap.summary_plot(sv, dense, feature_names=[clean_name(n) for n in names], max_display=20, show=False, plot_size=None); plt.title("Transaction XGBoost SHAP summary"); plt.tight_layout(); plt.savefig(FIG / "global_shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()
    # Perturbation around the robust-scaled median (zero) for a compact, model-agnostic check.
    top = imp.head(20).feature.tolist(); top_idx = [names.index(n) for n in top]; base = model.predict_proba(dense)[:, 1]; changes = []
    for j, name in zip(top_idx, top):
        altered = dense.copy(); altered[:, j] = 0.0; changed = model.predict_proba(altered)[:, 1]; changes.append({"feature": name, "mean_absolute_probability_change": float(np.abs(base - changed).mean())})
    pert = pd.DataFrame(changes).sort_values("mean_absolute_probability_change", ascending=False); pert.to_csv(RESULTS / "global_perturbation_importance.csv", index=False); bar(pert.mean_absolute_probability_change.to_numpy(), pert.feature.tolist(), "Transaction XGBoost perturbation importance", FIG / "global_perturbation_bar.png", color="#c05621", xlabel="Mean absolute probability change")
    # Six auditable local examples: three true positives and three true negatives.
    examples = []
    for label, candidates in [("tp", np.flatnonzero((y == 1) & (yh == 1))[:3]), ("tn", np.flatnonzero((y == 0) & (yh == 0))[:3])]:
        for k, row in enumerate(candidates, 1):
            one = X[row].toarray().astype(np.float32); one_sv = explainer.shap_values(one); one_sv = one_sv[0] if isinstance(one_sv, list) else np.asarray(one_sv); one_sv = one_sv[0]
            vals = one_sv; order = np.argsort(np.abs(vals))[-15:][::-1]; fig, ax = plt.subplots(figsize=(9, 6)); labels = [clean_name(names[i]) for i in order][::-1]; vv = vals[order][::-1]; colors = ["#c53030" if v > 0 else "#2b6cb0" for v in vv]; yy = np.arange(len(labels)); ax.barh(yy, vv, color=colors); ax.set_yticks(yy, labels); ax.axvline(0, color="k", lw=.8); ax.set_xlabel("SHAP contribution to laundering probability"); ax.set_title(f"{label.upper()} transaction explanation | row {row} | true={y[row]} | probability={p[row]:.4f}");
            for yi, v in zip(yy, vv): ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
            fig.tight_layout(); fig.savefig(FIG / f"{label}_{k}_shap.png", dpi=300, bbox_inches="tight"); plt.close(fig)
            examples.append({"example": f"{label}_{k}", "test_row": int(row), "Transaction_id": pred.iloc[row].Transaction_id, "Account_id": pred.iloc[row].Account_id, "true": int(y[row]), "prediction": int(yh[row]), "probability": float(p[row]), "largest_absolute_feature": clean_name(names[order[0]])})
    pd.DataFrame(examples).to_csv(RESULTS / "local_examples.csv", index=False)
    print("Saved explainability artefacts to", RESULTS)


if __name__ == "__main__": main()
