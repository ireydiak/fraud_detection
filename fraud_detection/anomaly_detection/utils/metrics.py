import numpy as np
from sklearn import metrics as sk_metrics


def evaluate(combined_score, test_score, test_labels, threshold, pos_label=1):
    thresh = np.percentile(combined_score, threshold)

    # Prediction using the threshold value
    y_pred = (test_score >= thresh).astype(int)
    y_true = test_labels.astype(int)

    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )

    return {"Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "AUROC": sk_metrics.roc_auc_score(y_true, test_score),
            "AUPR": sk_metrics.average_precision_score(y_true, test_score)}

