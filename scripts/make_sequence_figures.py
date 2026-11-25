from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, roc_auc_score

root = Path(__file__).resolve().parents[1]
experiment = os.environ.get('AML_SEQUENCE_EXPERIMENT', 'sequence')
results = root / 'results' / experiment
figures = results / 'figures'
figures.mkdir(parents=True, exist_ok=True)
pred = pd.read_csv(results / 'test_predictions.csv')
y, score, predicted = pred.y_true.to_numpy(), pred.probability.to_numpy(), pred.prediction.to_numpy()

cm = confusion_matrix(y, predicted, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, cmap='Blues')
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Laundering'], yticklabels=['Normal', 'Laundering'], xlabel='Predicted label', ylabel='True label', title='Account-level GRU confusion matrix')
for i in range(2):
    for j in range(2): ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=15, color='white' if cm[i, j] > cm.max()/2 else 'black')
fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(figures / 'confusion_matrix.png', dpi=300); plt.close(fig)

report = pd.DataFrame(classification_report(y, predicted, labels=[0, 1], target_names=['Normal', 'Laundering'], output_dict=True, zero_division=0)).T
values = report.loc[['Normal', 'Laundering'], ['precision', 'recall', 'f1-score']].to_numpy()
fig, ax = plt.subplots(figsize=(8, 3.8)); im = ax.imshow(values, cmap='YlGn', vmin=0, vmax=1)
ax.set(xticks=range(3), yticks=range(2), xticklabels=['Precision', 'Recall', 'F1-score'], yticklabels=['Normal', 'Laundering'], title='Account-level GRU classification report')
for i in range(2):
    for j in range(3): ax.text(j, i, f'{values[i, j]:.3f}', ha='center', va='center', fontsize=13)
fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(figures / 'classification_report.png', dpi=300); plt.close(fig)

# Full report (including accuracy, macro average and weighted average) as a
# paper-ready table figure, kept independent from the compact heatmap.
full = report.copy(); full.index = full.index.astype(str)
cols = ['precision', 'recall', 'f1-score', 'support']
table_values = [[f'{v:.3f}' if c != 'support' else f'{v:.0f}' for c, v in zip(cols, full.loc[idx, cols])] for idx in full.index]
fig, ax = plt.subplots(figsize=(8.5, 3.6)); ax.axis('off')
tbl = ax.table(cellText=table_values, rowLabels=list(full.index), colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
ax.set_title('Account-level GRU classification report (complete)', pad=12)
fig.tight_layout(); fig.savefig(figures / 'classification_report_full.png', dpi=300, bbox_inches='tight'); plt.close(fig)

fpr, tpr, _ = roc_curve(y, score)
fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f'ROC-AUC={roc_auc_score(y, score):.3f}'); ax.plot([0, 1], [0, 1], 'k--'); ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Account-level GRU ROC curve'); ax.legend(); fig.tight_layout(); fig.savefig(figures / 'roc_curve.png', dpi=300); plt.close(fig)

precision, recall, _ = precision_recall_curve(y, score)
fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(recall, precision, label='Precision–recall'); ax.axhline(.8, color='r', ls='--', label='Precision 0.80'); ax.axvline(.8, color='g', ls='--', label='Recall 0.80'); ax.set(xlabel='Recall', ylabel='Precision', title='Account-level GRU precision–recall curve'); ax.legend(); fig.tight_layout(); fig.savefig(figures / 'precision_recall_curve.png', dpi=300); plt.close(fig)

history = pd.read_csv(results / 'training_history.csv')
metric_specs = [('account_probability_loss', 'GRU training loss', 'Binary cross-entropy', 'loss'), ('account_probability_pr_auc', 'GRU training PR-AUC', 'PR-AUC', 'pr_auc'), ('account_probability_roc_auc', 'GRU training ROC-AUC', 'ROC-AUC', 'roc_auc'), ('account_probability_precision', 'GRU training precision', 'Precision', 'precision'), ('account_probability_recall', 'GRU training recall', 'Recall', 'recall')]
for source_metric, title, ylabel, output_name in metric_specs:
    if source_metric not in history: continue
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(history[source_metric], marker='o', label='train'); val = 'val_' + source_metric
    if val in history: ax.plot(history[val], marker='o', label='validation')
    ax.set(xlabel='Epoch', ylabel=ylabel, title=title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / f'{output_name}.png', dpi=300); plt.close(fig)

print(f'Saved {len(list(figures.glob("*.png")))} independent figures to {figures}')
