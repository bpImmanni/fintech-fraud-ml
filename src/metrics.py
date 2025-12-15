import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

def compute_threshold_metrics(y_true, y_prob, threshold: float, review_cost=2.0, avg_fraud_loss=200.0):
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    est_savings = tp * avg_fraud_loss - fp * review_cost  # simple business proxy

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "est_savings": est_savings
    }

def recall_at_precision(y_true, y_prob, target_precision=0.95):
    p, r, t = precision_recall_curve(y_true, y_prob)
    # p/r arrays include an extra element; thresholds align to t
    # choose the best recall where precision >= target
    mask = p[:-1] >= target_precision
    if not mask.any():
        return {"threshold": None, "recall": 0.0, "precision": float(p.max())}
    idx = np.argmax(r[:-1][mask])
    thresh = t[mask][idx]
    return {"threshold": float(thresh), "recall": float(r[:-1][mask][idx]), "precision": float(p[:-1][mask][idx])}

def headline_metrics(y_true, y_prob):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
