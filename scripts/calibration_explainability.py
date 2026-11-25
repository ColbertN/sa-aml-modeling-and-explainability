"""Calibration and account-level explanations for the saved GRU model.

The script never refits the GRU.  Calibration parameters are learned from the
validation accounts and then evaluated on the frozen test accounts.  The
explanations use gradient-times-input saliency on the same padded sequences
used by the model; padding positions are removed before aggregation.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import tensorflow as tf
from train_sequence_from_scratch import MaskedMaxPool, MaskedMeanPool, MaskedAttentionPool
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sequence"
EXPERIMENT = os.environ.get("AML_SEQUENCE_EXPERIMENT", "sequence")
RESULTS = ROOT / "results" / EXPERIMENT
FIGURES = RESULTS / "figures"
MODEL_DIR = ROOT / "models" / ("sequence_gru" if EXPERIMENT == "sequence" else EXPERIMENT)
PAD = -1e6
EPS = 1e-6

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.titlesize": 13})


def pretty_feature(name: str) -> str:
    """Use readable labels without changing the persisted feature names."""
    return textwrap.fill(str(name).replace("_", " "), width=24)


def sparse_event_ticks(event_numbers, max_ticks=12):
    """Return positions and labels that remain readable for long histories."""
    event_numbers = np.asarray(event_numbers, dtype=int)
    if len(event_numbers) <= max_ticks:
        pos = np.arange(len(event_numbers))
    else:
        pos = np.unique(np.linspace(0, len(event_numbers) - 1, max_ticks).round().astype(int))
    return pos, [str(int(event_numbers[i])) for i in pos]


def logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def beta_features(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.column_stack([np.log(p), np.log1p(-p)])


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    value = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def plot_with_labels(path: Path, x, y, title: str, ylabel: str, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(x, y, color="#3568a8")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
    if ylim: ax.set_ylim(*ylim)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + (0.01 if ylim and ylim[1] <= 1 else 0),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def full_classification_report_figure(pred: pd.DataFrame):
    from sklearn.metrics import classification_report
    report = pd.DataFrame(classification_report(
        pred.y_true, pred.prediction, labels=[0, 1],
        target_names=["Normal", "Laundering"], output_dict=True, zero_division=0
    )).T
    order = ["Normal", "Laundering", "accuracy", "macro avg", "weighted avg"]
    report = report.reindex(order)
    # sklearn represents accuracy as a scalar; make the displayed table match
    # the conventional classification-report layout (one accuracy value and
    # the total test support, rather than duplicated values and support=1).
    total_support = int(report.loc[["Normal", "Laundering"], "support"].sum())
    accuracy_value = float(report.loc["accuracy", "f1-score"])
    report.loc["accuracy", ["precision", "recall"]] = np.nan
    report.loc["accuracy", "f1-score"] = accuracy_value
    report.loc["accuracy", "support"] = total_support
    report.to_csv(RESULTS / "classification_report.csv")
    cols = ["precision", "recall", "f1-score", "support"]
    cell = []
    for idx, row in report.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append("—" if pd.isna(v) else (f"{v:.3f}" if c != "support" else f"{int(round(v)):,}"))
        display_name = {"macro avg": "Macro average", "weighted avg": "Weighted average"}.get(idx, idx.title() if idx == "accuracy" else idx)
        cell.append([display_name] + vals)
    fig, ax = plt.subplots(figsize=(9.5, 3.5)); ax.axis("off")
    table = ax.table(cellText=cell, colLabels=["Class / summary"] + ["Precision", "Recall", "F1-score", "Support"],
                     cellLoc="center", colLoc="center", loc="center", colWidths=[.25, .18, .18, .18, .18])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.65)
    for (r, c), ce in table.get_celld().items():
        ce.set_edgecolor("#b7c3d0")
        if r == 0: ce.set_facecolor("#244a73"); ce.set_text_props(color="white", weight="bold")
        elif r in (1, 2): ce.set_facecolor("#eef5fb")
        elif r in (3, 4, 5): ce.set_facecolor("#f7f7f7")
    ax.set_title("Account-level GRU classification report (frozen test set)", pad=12, weight="bold")
    fig.tight_layout()
    # Keep the concise, conventional filename pointing to the complete table
    # as well; this prevents an older two-row report from being mistaken for
    # the current result.
    fig.savefig(FIGURES / "classification_report_full.png", bbox_inches="tight")
    fig.savefig(FIGURES / "classification_report.png", bbox_inches="tight")
    plt.close(fig)


def fit_calibrators(yv: np.ndarray, pv: np.ndarray):
    lv = logits(pv)
    platt = LogisticRegression(solver="lbfgs").fit(lv.reshape(-1, 1), yv)
    beta = LogisticRegression(solver="lbfgs").fit(beta_features(pv), yv)

    def objective(temp):
        return log_loss(yv, sigmoid(lv / temp), labels=[0, 1])
    opt = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    temperature = float(opt.x)
    return platt, beta, temperature


def calibrate(p: np.ndarray, platt, beta, temperature: float):
    l = logits(p)
    return {
        "uncalibrated": np.clip(p, EPS, 1-EPS),
        "platt": platt.predict_proba(l.reshape(-1, 1))[:, 1],
        "beta": beta.predict_proba(beta_features(p))[:, 1],
        "temperature": sigmoid(l / temperature),
    }


def calibration_outputs(pred: pd.DataFrame):
    yv = np.load(DATA / "y_validation.npy").astype(int)
    yt = pred.y_true.to_numpy(int)
    model = tf.keras.models.load_model(MODEL_DIR / "model.keras", compile=False)
    Xv = [np.load(DATA / name, mmap_mode="r") for name in ("X_validation.npy", "X_validation_inbound.npy", "X_validation_outbound.npy")]
    raw = model.predict(Xv, batch_size=512, verbose=0)
    pv = (raw[0] if isinstance(raw, (list, tuple)) else raw).ravel()
    pt = pred.probability.to_numpy(float)
    platt, beta, temperature = fit_calibrators(yv, pv)
    test_probs = calibrate(pt, platt, beta, temperature)
    val_probs = calibrate(pv, platt, beta, temperature)
    rows = []
    for name, probs in test_probs.items():
        rows.append({"method": name, "brier_score": brier_score_loss(yt, probs),
                     "log_loss": log_loss(yt, probs, labels=[0, 1]), "ece_10_bins": ece(yt, probs)})
    pd.DataFrame(rows).to_csv(RESULTS / "calibration_metrics.csv", index=False)
    out = pred.copy()
    for name, probs in test_probs.items(): out[f"probability_{name}"] = probs
    out.to_csv(RESULTS / "calibration_test_predictions.csv", index=False)
    joblib.dump({"platt": platt, "beta": beta, "temperature": temperature, "methods": ["uncalibrated", "platt", "beta", "temperature"]}, MODEL_DIR / "calibrators.joblib")
    (MODEL_DIR / "calibration_manifest.json").write_text(json.dumps({"fit_split": "validation", "evaluation_split": "test", "isotonic_used": False, "temperature": temperature}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 5.5)); ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    colors = {"uncalibrated": "#2878b5", "platt": "#e67e22", "beta": "#2ca25f", "temperature": "#8e44ad"}
    for name, probs in test_probs.items():
        frac, mean = calibration_curve(yt, probs, n_bins=10, strategy="uniform")
        ax.plot(mean, frac, marker="o", lw=1.8, color=colors[name], label=name)
        for x, y in zip(mean, frac): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=colors[name])
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Account-level GRU calibration (10 bins)"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration_curves.png", bbox_inches="tight"); plt.close(fig)
    metrics = pd.DataFrame(rows)
    for metric in ["brier_score", "log_loss", "ece_10_bins"]:
        plot_with_labels(FIGURES / f"calibration_{metric}.png", metrics.method, metrics[metric], f"Calibration comparison: {metric.replace('_', ' ').title()}", metric.replace('_', ' ').title())
    return model


def explain_accounts(model, pred: pd.DataFrame):
    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    names = joblib.load(MODEL_DIR / "preprocessing.joblib")["feature_names"]; seq_len = int(manifest["sequence_length"])
    X = np.load(DATA / "X_test.npy", mmap_mode="r")
    X_in = np.load(DATA / "X_test_inbound.npy", mmap_mode="r")
    X_out = np.load(DATA / "X_test_outbound.npy", mmap_mode="r")
    y = pred.y_true.to_numpy(int); p = pred.probability.to_numpy(float); yp = pred.prediction.to_numpy(int)
    tp = np.where((y == 1) & (yp == 1))[0]; tn = np.where((y == 0) & (yp == 0))[0]
    selected = list(tp[np.argsort(-p[tp])[:3]]) + list(tn[np.argsort(p[tn])[:3]])
    labels = ["tp_1", "tp_2", "tp_3", "tn_1", "tn_2", "tn_3"]
    # Differentiate the pre-sigmoid score.  Several very confident accounts
    # have sigmoid output rounded to 1 or 0; differentiating that saturated
    # probability would produce an uninformative all-zero heatmap.
    head = model.get_layer("account_probability")
    representation = tf.keras.Model(model.inputs, head.input)

    def pre_score(batch):
        hidden = representation([tf.convert_to_tensor(z, dtype=tf.float32) for z in batch], training=False)
        return (tf.linalg.matmul(hidden, head.kernel) + head.bias).numpy().ravel()
    summary = []
    for tag, idx in zip(labels, selected):
        arr = np.asarray(X[idx:idx+1], dtype="float32")
        arr_in = np.asarray(X_in[idx:idx+1], dtype="float32")
        arr_out = np.asarray(X_out[idx:idx+1], dtype="float32")
        inp = [tf.convert_to_tensor(z) for z in (arr, arr_in, arr_out)]
        with tf.GradientTape() as tape:
            tape.watch(inp[0])
            hidden = representation(inp, training=False)
            score = tf.linalg.matmul(hidden, head.kernel) + head.bias
        grad = tape.gradient(score, inp[0]).numpy()[0]
        sal = np.abs(grad * arr[0]); pad_mask = np.all(np.isclose(arr[0], PAD), axis=1); sal[pad_mask] = 0
        event_imp = sal.sum(axis=1); feature_imp = sal.sum(axis=0)
        valid_events = np.where(~pad_mask)[0]
        if len(valid_events) == 0: valid_events = np.arange(seq_len)
        top_features = np.argsort(-feature_imp)[:12]
        top_events = np.argsort(-event_imp)[:min(12, seq_len)]
        # Feature importance (one account per figure)
        fig, ax = plt.subplots(figsize=(8.5, 5)); vals = feature_imp[top_features][::-1]; labs = [names[i] for i in top_features[::-1]]
        bars = ax.barh(labs, vals, color="#3b79a8"); ax.set_title(f"{('True positive' if y[idx] else 'True negative')} account {tag[-1]} — feature importance\nAccount_id={pred.Account_id.iloc[idx]}"); ax.set_xlabel("Gradient × input magnitude"); ax.grid(axis="x", alpha=.2)
        for b, v in zip(bars, vals): ax.text(v, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_feature_importance.png", bbox_inches="tight"); plt.close(fig)
        # Event importance (top events, with chronological event index labels)
        ev = top_events[np.argsort(event_imp[top_events])]; vals = event_imp[ev]
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); bars = ax.bar([f"event {i+1}" for i in ev], vals, color="#c56b32"); ax.set_title(f"{('True positive' if y[idx] else 'True negative')} account {tag[-1]} — event importance"); ax.set_ylabel("Aggregated saliency"); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=.2)
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_event_importance.png", bbox_inches="tight"); plt.close(fig)
        # Event × feature heatmap. Annotating the compact 12-feature matrix keeps labels readable.
        mat = sal[:, top_features]; mat = mat / (mat.max() + EPS)
        fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1); ax.set_xticks(range(len(top_features)), [names[i] for i in top_features], rotation=70, ha="right"); ax.set_yticks(range(seq_len), [f"{i+1}" for i in range(seq_len)]); ax.set(xlabel="Top features", ylabel="Chronological event", title=f"{('True positive' if y[idx] else 'True negative')} account {tag[-1]} — event × feature contribution"); fig.colorbar(im, ax=ax, label="Relative saliency")
        for r in range(seq_len):
            for c in range(len(top_features)):
                # Every cell is labelled so the figure remains interpretable
                # when it is separated from the manuscript text.
                colour = "white" if mat[r, c] >= .45 else "#f5d7a1"
                ax.text(c, r, f"{mat[r,c]:.1f}", ha="center", va="center", fontsize=4, color=colour)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_event_feature_heatmap.png", bbox_inches="tight"); plt.close(fig)
        # Faithful perturbation checks: replace one feature (or one event) by
        # the training-centre value zero and measure the change in the model's
        # pre-sigmoid score.  This complements gradient saliency and is easy
        # to explain in a methods section.
        base = float(pre_score([arr, arr_in, arr_out])[0]); n_features = arr.shape[2]
        feature_batch = np.repeat(arr, n_features, axis=0)
        for f in range(n_features): feature_batch[f, ~pad_mask, f] = 0.0
        perturb_feature = np.abs(base - pre_score([feature_batch, np.repeat(arr_in, n_features, axis=0), np.repeat(arr_out, n_features, axis=0)]))
        event_batch = np.repeat(arr, seq_len, axis=0)
        for e in range(seq_len):
            if not pad_mask[e]: event_batch[e, e, :] = 0.0
        perturb_event = np.abs(base - pre_score([event_batch, np.repeat(arr_in, seq_len, axis=0), np.repeat(arr_out, seq_len, axis=0)]))
        pf = np.argsort(-perturb_feature)[:12][::-1]; pe = np.argsort(-perturb_event)[:min(12, seq_len)][::-1]
        fig, ax = plt.subplots(figsize=(8.5, 5)); vals = perturb_feature[pf]; bars = ax.barh([names[i] for i in pf], vals, color="#6b4c9a"); ax.set_title(f"{('True positive' if y[idx] else 'True negative')} account {tag[-1]} — perturbation feature importance"); ax.set_xlabel("Absolute change in pre-sigmoid score"); ax.grid(axis="x", alpha=.2)
        for b, v in zip(bars, vals): ax.text(v, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_perturbation_feature_importance.png", bbox_inches="tight"); plt.close(fig)
        summary.append({"example": tag, "Account_id": pred.Account_id.iloc[idx], "test_row": int(idx), "true_label": int(y[idx]), "probability": float(p[idx]), "prediction": int(yp[idx]), "top_saliency_feature": names[int(top_features[0])], "top_saliency_event": int(top_events[0] + 1), "top_perturbation_feature": names[int(pf[-1])], "top_perturbation_event": int(pe[-1] + 1), "valid_events": int((~pad_mask).sum())})
    pd.DataFrame(summary).to_csv(RESULTS / "explainability_examples.csv", index=False)


def explain_accounts_clean(model, pred: pd.DataFrame):
    """Generate readable local explanations with padding removed from axes."""
    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    names = joblib.load(MODEL_DIR / "preprocessing.joblib")["feature_names"]
    seq_len = int(manifest["sequence_length"])
    X = np.load(DATA / "X_test.npy", mmap_mode="r")
    X_in = np.load(DATA / "X_test_inbound.npy", mmap_mode="r")
    X_out = np.load(DATA / "X_test_outbound.npy", mmap_mode="r")
    y = pred.y_true.to_numpy(int); p = pred.probability.to_numpy(float); yp = pred.prediction.to_numpy(int)
    tp = np.where((y == 1) & (yp == 1))[0]; tn = np.where((y == 0) & (yp == 0))[0]
    selected = list(tp[np.argsort(-p[tp])[:3]]) + list(tn[np.argsort(p[tn])[:3]])
    labels = ["tp_1", "tp_2", "tp_3", "tn_1", "tn_2", "tn_3"]
    head = model.get_layer("account_probability")
    representation = tf.keras.Model(model.inputs, head.input)

    def pre_score(batch):
        hidden = representation([tf.convert_to_tensor(z, dtype=tf.float32) for z in batch], training=False)
        return (tf.linalg.matmul(hidden, head.kernel) + head.bias).numpy().ravel()

    summary = []
    for tag, idx in zip(labels, selected):
        arr = np.asarray(X[idx:idx + 1], dtype="float32")
        arr_in = np.asarray(X_in[idx:idx + 1], dtype="float32")
        arr_out = np.asarray(X_out[idx:idx + 1], dtype="float32")
        pad_mask = np.all(np.isclose(arr[0], PAD), axis=1)
        valid_events = np.where(~pad_mask)[0]
        if len(valid_events) == 0:
            valid_events = np.arange(seq_len)
        with tf.GradientTape() as tape:
            tensor = tf.convert_to_tensor(arr)
            tape.watch(tensor)
            hidden = representation([tensor, tf.convert_to_tensor(arr_in), tf.convert_to_tensor(arr_out)], training=False)
            score = tf.linalg.matmul(hidden, head.kernel) + head.bias
        signed = tape.gradient(score, tensor).numpy()[0] * arr[0]
        signed[pad_mask] = 0
        sal = np.abs(signed)
        event_imp = sal.sum(axis=1); feature_imp = sal.sum(axis=0)
        top_features = np.argsort(-feature_imp)[:12]
        top_events = valid_events[np.argsort(-event_imp[valid_events])[:min(12, len(valid_events))]]
        kind = "True positive" if y[idx] else "True negative"
        account = pred.Account_id.iloc[idx]

        vals = feature_imp[top_features][::-1]
        bars = None
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        bars = ax.barh([pretty_feature(names[i]) for i in top_features[::-1]], vals, color="#3568a8")
        ax.set_title(f"{kind} account {tag[-1]} - feature importance\nAccount ID={account}")
        ax.set_xlabel("Absolute gradient x input contribution"); ax.grid(axis="x", alpha=.2)
        for b, v in zip(bars, vals): ax.text(v, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_feature_importance.png", bbox_inches="tight"); plt.close(fig)

        ev = top_events[np.argsort(event_imp[top_events])]; vals = event_imp[ev]
        fig, ax = plt.subplots(figsize=(8.4, max(4.2, .32 * len(ev) + 1.6)))
        bars = ax.barh([f"Event {i + 1}" for i in ev], vals, color="#d95f59")
        ax.set_title(f"{kind} account {tag[-1]} - event importance\nAccount ID={account}")
        ax.set_xlabel("Absolute contribution"); ax.grid(axis="x", alpha=.2)
        for b, v in zip(bars, vals): ax.text(v, b.get_y() + b.get_height() / 2, f"{v:.2f}", ha="left", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_event_importance.png", bbox_inches="tight"); plt.close(fig)

        mat = signed[valid_events][:, top_features].T
        limit = max(float(np.abs(mat).max()), EPS)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        event_numbers = valid_events + 1
        x_pos, x_labels = sparse_event_ticks(event_numbers)
        width = max(10.0, min(18.0, 7.0 + .10 * len(valid_events)))
        fig, ax = plt.subplots(figsize=(width, 5.8))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_xticks(x_pos, x_labels)
        ax.set_yticks(range(len(top_features)), [pretty_feature(names[i]) for i in top_features])
        ax.set(xlabel="Observed transaction number (chronological)", ylabel="Top features",
               title=f"{kind} account {tag[-1]} - event by feature contribution\nAccount ID={account}")
        fig.colorbar(im, ax=ax, label="Signed contribution (red increases, blue decreases)")
        # One label per feature row keeps the strongest evidence visible
        # without placing adjacent numbers on top of one another.
        for rr in range(mat.shape[0]):
            cc = int(np.argmax(np.abs(mat[rr])))
            value = mat[rr, cc]
            if abs(value) <= 0.02 * limit:
                continue
            text_color = "white" if abs(value) > .55 * limit else "black"
            ax.text(cc, rr, f"{value:+.2f}", ha="center", va="center", fontsize=7, color=text_color)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_event_feature_heatmap.png", bbox_inches="tight"); plt.close(fig)

        base = float(pre_score([arr, arr_in, arr_out])[0]); n_features = arr.shape[2]
        feature_batch = np.repeat(arr, n_features, axis=0)
        for f in range(n_features): feature_batch[f, ~pad_mask, f] = 0.0
        perturb_feature = np.abs(base - pre_score([feature_batch, np.repeat(arr_in, n_features, axis=0), np.repeat(arr_out, n_features, axis=0)]))
        pf = np.argsort(-perturb_feature)[:12][::-1]
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        vals = perturb_feature[pf]
        bars = ax.barh([pretty_feature(names[i]) for i in pf], vals, color="#6b4c9a")
        ax.set_title(f"{kind} account {tag[-1]} - perturbation feature importance\nAccount ID={account}")
        ax.set_xlabel("Absolute change in pre-sigmoid score"); ax.grid(axis="x", alpha=.2)
        for b, v in zip(bars, vals): ax.text(v, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_perturbation_feature_importance.png", bbox_inches="tight"); plt.close(fig)
        summary.append({"example": tag, "Account_id": account, "test_row": int(idx), "true_label": int(y[idx]), "probability": float(p[idx]), "prediction": int(yp[idx]), "top_saliency_feature": names[int(top_features[0])], "top_saliency_event": int(top_events[0] + 1), "top_perturbation_feature": names[int(pf[-1])], "valid_events": int(len(valid_events))})
    pd.DataFrame(summary).to_csv(RESULTS / "explainability_examples.csv", index=False)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    full_classification_report_figure(pred)
    model = calibration_outputs(pred)
    explain_accounts_clean(model, pred)
    print("Saved calibration metrics, calibrators, full report, and six account explanation sets.")


if __name__ == "__main__":
    main()
