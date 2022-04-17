import numpy as np


def precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 0):
    tp = len([a for a, p in zip(y_true, y_pred) if a == p and p == pos_label])
    fp = len([a for a, p in zip(y_true, y_pred) if a != p and p == pos_label])
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 0):
    tp = len([a for a, p in zip(y_true, y_pred) if a == p and p == pos_label])
    fn = len([a for a, p in zip(y_true, y_pred) if a != p and p == int(not bool(pos_label))])
    return tp / (tp + fn)


def f1score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 0):
    prec = precision(y_true, y_pred, pos_label)
    rec = recall(y_true, y_pred, pos_label)
    return (2 * prec * rec) / (rec + prec)


def precision_recall_f1score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 0):
    return precision(y_true, y_pred, pos_label), recall(y_true, y_pred, pos_label), f1score(y_true, y_pred, pos_label)
