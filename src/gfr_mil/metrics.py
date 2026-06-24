from __future__ import annotations

import math

import numpy as np


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return wins / (len(pos) * len(neg))


def _macro_f1(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        tp = float(np.sum((labels == class_id) & (preds == class_id)))
        fp = float(np.sum((labels != class_id) & (preds == class_id)))
        fn = float(np.sum((labels == class_id) & (preds != class_id)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores))


def classification_metrics(
    labels: list[int] | np.ndarray,
    probs: list[list[float]] | np.ndarray,
    preds: list[int] | np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = np.asarray(preds, dtype=int)

    metrics = {
        "acc": float(np.mean(labels == preds)) if len(labels) else 0.0,
        "f1_macro": _macro_f1(labels, preds, num_classes) if len(labels) else 0.0,
    }
    if len(labels) == 0:
        metrics["auc"] = float("nan")
        return metrics

    if num_classes == 2:
        metrics["auc"] = _binary_auc(labels, probs[:, 1])
    else:
        aucs = []
        for class_id in range(num_classes):
            binary_labels = (labels == class_id).astype(int)
            aucs.append(_binary_auc(binary_labels, probs[:, class_id]))
        valid = [value for value in aucs if not math.isnan(value)]
        metrics["auc"] = float(np.mean(valid)) if valid else float("nan")
    return metrics
