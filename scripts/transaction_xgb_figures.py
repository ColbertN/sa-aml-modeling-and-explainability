"""Create publication-ready, independent evaluation figures for transaction XGBoost."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score, average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "transaction_xgb"
OUT = RESULTS / "figures"


def save_report_figure(report, path):
    rows = ["Normal", "Laundering", "Accuracy", "Macro average", "Weighted average"]
    cols = ["precision", "recall", "f1-score", "support"]
    table = pd.DataFrame(index=rows, columns=cols)
    table.loc["Normal"] = [report["0"][c] for c in ["precision", "recall", "f1-score", "support"]]
    table.loc["Laundering"] = [report["1"][c] for c in ["precision", "recall", "f1-score", "support"]]
    table.loc["Accuracy"] = [np.nan, np.nan, report["accuracy"], report["0"]["support"] + report["1"]["support"]]
    table.loc["Macro average"] = [report["macro avg"][c] for c in cols]
    table.loc["Weighted average"] = [report["weighted avg"][c] for c in cols]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")
    cell_text = [["—" if pd.isna(v) else (f"{v:.3f}" if c != "support" else f"{int(v):,}") for c, v in zip(cols, table.loc[r])] for r in rows]
    tab = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, cellLoc="center", rowLoc="center", loc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.6)
    for (r, c), cell in tab.get_celld().items():
        if r == 0: cell.set_facecolor("#1f4e79"); cell.set_text_props(color="white", weight="bold")
        elif r in (1, 2): cell.set_facecolor("#eaf2f8")
        elif r == 3: cell.set_facecolor("#d9ead3")
    ax.set_title("Transaction-level XGBoost classification report", pad=12, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    y = pred.y_true.to_numpy(dtype=int); p = pred.probability.to_numpy(dtype=float); threshold = float(pred.attrs.get("threshold", np.nan))
    metrics = pd.read_csv(RESULTS / "test_metrics.csv").iloc[0]
    threshold = float(metrics.threshold)
    pred_y = (p >= threshold).astype(int)
    report = classification_report(y, pred_y, output_dict=True, zero_division=0)
    save_report_figure(report, OUT / "classification_report.png")
    cm = confusion_matrix(y, pred_y)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Laundering"]); disp.plot(ax=ax, cmap="Blues", values_format=",d", colorbar=False); ax.set_title(f"Transaction-level XGBoost confusion matrix (threshold={threshold:.3f})"); fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=300); plt.close(fig)
    fpr, tpr, _ = roc_curve(y, p); ra = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(6.5, 5.2)); ax.plot(fpr, tpr, label=f"XGBoost (ROC-AUC={ra:.3f})", lw=2); ax.plot([0,1],[0,1], "k--", label="Chance"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Transaction-level XGBoost ROC curve"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=300); plt.close(fig)
    prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(6.5, 5.2)); ax.plot(rec, prec, label=f"XGBoost (PR-AUC={ap:.3f})", lw=2); ax.axhline(y.mean(), color="k", ls="--", label=f"Prevalence={y.mean():.3f}"); ax.set(xlabel="Recall", ylabel="Precision", title="Transaction-level XGBoost precision–recall curve"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "pr_curve.png", dpi=300); plt.close(fig)
    tc = pd.read_csv(RESULTS / "threshold_candidates.csv")
    cols = [c for c in ["threshold", "precision", "recall", "f1", "predicted_positive_rate"] if c in tc]
    fig, ax = plt.subplots(figsize=(7, 4.5));
    for c in cols[1:]: ax.plot(tc["threshold"], tc[c], marker=".", label=c)
    ax.axvline(threshold, color="k", ls="--", label=f"selected={threshold:.3f}"); ax.set(xlabel="Decision threshold", ylabel="Score", title="Transaction-level threshold analysis"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "threshold_analysis.png", dpi=300); plt.close(fig)
    print(pd.DataFrame(report).T.to_string())


if __name__ == "__main__": main()
