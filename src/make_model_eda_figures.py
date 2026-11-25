"""Create clear, independent exploratory figures for the modelling paper.

The figures use raw amounts (not a logarithmic axis), place sample sizes in the
legend rather than over the curves, and add labels without creating crowded
annotations.  Only public input columns and the transaction label are read.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "SA_SAML_D.csv"
OUT = ROOT / "paper" / "Journal_2_SA_AML_Modelling" / "figures"
SEED = 42

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    usecols = [
        "Is_laundering", "Account_id", "Amount", "Payment_type", "Direction",
        "Date", "Time", "Payment_currency", "Received_currency",
        "Receiver_bank_location", "Counterparty_type",
    ]
    df = pd.read_csv(RAW, usecols=usecols, low_memory=False)
    df["Is_laundering"] = pd.to_numeric(df["Is_laundering"], errors="coerce").fillna(0).astype(int)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).clip(lower=0)

    # Account- and transaction-level prevalence.
    account = df.groupby("Account_id", as_index=False)["Is_laundering"].max()
    account_counts = account["Is_laundering"].value_counts().reindex([0, 1], fill_value=0)
    txn_counts = df["Is_laundering"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x = np.arange(2); width = .36
    b1 = ax.bar(x - width / 2, account_counts.values, width, label="Accounts", color="#3568a8")
    b2 = ax.bar(x + width / 2, txn_counts.values, width, label="Transactions", color="#d95f59")
    ax.set_xticks(x, ["Normal", "Laundering"])
    ax.set_ylabel("Count")
    ax.set_title("Class prevalence before modelling")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.22)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", fontsize=8)
    save(fig, "mod_eda_prevalence.png")

    # Amount ECDF on the original currency scale.  The curves are sampled only
    # for rendering; the displayed n values are the full class counts.
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {0: "#3568a8", 1: "#d95f59"}
    names = {0: "Normal", 1: "Laundering"}
    for label in (0, 1):
        vals = df.loc[df.Is_laundering == label, "Amount"].to_numpy(float)
        sample = vals if len(vals) <= 100_000 else rng.choice(vals, 100_000, replace=False)
        sample.sort(); yy = np.arange(1, len(sample) + 1) / len(sample)
        ax.plot(sample, yy, lw=2.0, color=colors[label], label=f"{names[label]} (n={len(vals):,})")
    display_limit = float(np.quantile(df["Amount"], .995))
    ax.set_xlim(0, display_limit)
    ax.set_xlabel("Transaction amount (raw scale; shown to 99.5th percentile)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Transaction amount overlap by class")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(alpha=.22)
    save(fig, "mod_eda_amount_ecdf.png")

    # Payment overlap: within-class shares, with totals at the end of each row.
    top_types = df["Payment_type"].value_counts().head(8).index.tolist()
    ctab = pd.crosstab(df["Payment_type"], df["Is_laundering"]).reindex(top_types).fillna(0)
    shares = ctab.div(ctab.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    left = np.zeros(len(top_types))
    for label, color, name in [(0, "#3568a8", "Normal"), (1, "#d95f59", "Laundering")]:
        vals = shares[label].to_numpy()
        bars = ax.barh(top_types, vals, left=left, color=color, label=name)
        for bar, val in zip(bars, vals):
            if val >= .035:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        f"{100 * val:.1f}%", ha="center", va="center", color="white", fontsize=8)
        left += vals
    totals = ctab.sum(axis=1).to_numpy()
    for i, total in enumerate(totals):
        ax.text(1.005, i, f"n={int(total):,}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Within-class share")
    ax.set_title("Payment-type composition and overlap")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=.22)
    save(fig, "mod_eda_payment_overlap.png")

    # Account activity distribution with sample-size labels.
    activity = df.groupby("Account_id").agg(
        transaction_count=("Amount", "size"), account_label=("Is_laundering", "max")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = [activity.loc[activity.account_label == label, "transaction_count"] for label in (0, 1)]
    bp = ax.boxplot(data, labels=["Normal", "Laundering"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#9ecae1", "#f4a582"]): patch.set_facecolor(color)
    for x, vals in zip([1, 2], data):
        ax.text(x, np.nanmax(vals), f"n={len(vals):,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Transactions per account")
    ax.set_title("Account activity distribution by class")
    ax.grid(axis="y", alpha=.22)
    save(fig, "mod_eda_activity_boxplot.png")

    # Direction composition with within-label percentages.
    dtab = pd.crosstab(df["Direction"], df["Is_laundering"]).reindex(columns=[0, 1], fill_value=0)
    dshare = dtab.div(dtab.sum(axis=0), axis=1).fillna(0)
    directions = dshare.index.tolist(); x = np.arange(len(directions)); width = .36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars0 = ax.bar(x - width / 2, dshare[0], width, label="Normal", color="#3568a8")
    bars1 = ax.bar(x + width / 2, dshare[1], width, label="Laundering", color="#d95f59")
    ax.set_xticks(x, directions); ax.set_ylim(0, 1.05); ax.set_ylabel("Within-class share")
    ax.set_title("Transaction direction composition by class"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.22)
    for bars in (bars0, bars1):
        for bar in bars: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8)
    save(fig, "mod_eda_direction_composition.png")

    # Compact correlation diagnostic, with numeric values in every cell.
    numeric = df[["Amount", "Is_laundering"]].copy()
    numeric["currency_mismatch"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)
    numeric["cross_border"] = (df["Receiver_bank_location"].astype(str) != "South Africa").astype(int)
    numeric["account_transactions"] = df.groupby("Account_id")["Amount"].transform("size")
    corr = numeric.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                linewidths=.5, linecolor="white", cbar_kws={"label": "Spearman correlation"})
    ax.set_title("Spearman multicollinearity diagnostic")
    save(fig, "mod_eda_multicollinearity.png")


if __name__ == "__main__":
    main()
