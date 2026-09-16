"""Evaluation metrics for segmentation, CDR regression and classification."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score, roc_auc_score, roc_curve)


def dice(pred, gt, eps=1e-7):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / (pred.sum() + gt.sum() + eps))


def iou(pred, gt, eps=1e-7):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    return float((pred & gt).sum() / ((pred | gt).sum() + eps))


def regression_metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return {
        "mean_pred": float(pred.mean()),
        "mean_true": float(true.mean()),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)) if len(true) > 1 else float("nan"),
        "pearson_r": float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else float("nan"),
    }


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    i = int(np.argmax(tpr - fpr))
    return float(min(max(thr[i], 0.0), 1.0))


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else float("nan")
    return out
