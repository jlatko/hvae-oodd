import sklearn.metrics
import numpy as np

# Helper methods
def collapse_multiclass_to_binary(y_true, zero_label=None):
    # Force the class index in zero_label to be zero and the others to collapse to 1
    zero_label_indices = y_true == zero_label
    y_true[zero_label_indices] = 0
    y_true[~zero_label_indices] = 1
    return y_true


def compute_roc_auc(y_true=None, y_score=None, zero_label=None):
    """Only binary"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score, average="macro")
    return roc_auc, fpr, tpr, thresholds


def compute_pr_auc(y_true=None, y_score=None, zero_label=None):
    """Only binary"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score, average="macro")
    return pr_auc, precision, recall, thresholds


def compute_roc_pr_metrics(y_true, y_score, reference_class):
    roc_auc, fpr, tpr, thresholds = compute_roc_auc(y_true=y_true, y_score=y_score, zero_label=reference_class)
    pr_auc, precision, recall, thresholds = compute_pr_auc(y_true=y_true, y_score=y_score, zero_label=reference_class)
    idx_where_tpr_is_eighty = np.where((tpr - 0.8 >= 0))[0][0]
    fpr80 = fpr[idx_where_tpr_is_eighty]
    return (roc_auc, fpr, tpr, thresholds), (pr_auc, precision, recall, thresholds), fpr80