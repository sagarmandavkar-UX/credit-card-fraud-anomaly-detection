from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_transactions(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {*FEATURE_COLUMNS, "Class"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required transaction fields cannot be missing")
    if not data["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    return data.sort_values("Time", kind="stable").reset_index(drop=True)


def time_ordered_split(data: pd.DataFrame, train_fraction: float = 0.60, validation_fraction: float = 0.20):
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition")
    train_end = int(len(data) * train_fraction)
    validation_end = int(len(data) * (train_fraction + validation_fraction))
    return data.iloc[:train_end].copy(), data.iloc[train_end:validation_end].copy(), data.iloc[validation_end:].copy()


def alert_threshold(scores: np.ndarray, alert_rate: float) -> float:
    if not 0 < alert_rate < 1:
        raise ValueError("alert_rate must be between 0 and 1")
    return float(np.quantile(scores, 1 - alert_rate))


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    predictions = np.asarray(scores) >= threshold
    alerts = int(predictions.sum())
    frauds = int(truth.sum())
    captured = int(truth[predictions].sum())
    return {
        "transactions": int(len(truth)),
        "frauds": frauds,
        "alerts": alerts,
        "frauds_captured": captured,
        "alert_rate": alerts / len(truth),
        "pr_auc": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "false_positives": int(((truth == 0) & predictions).sum()),
    }


def review_curve(y_true: np.ndarray, scores: np.ndarray, review_rates: list[float]) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=int)
    order = np.argsort(scores)[::-1]
    rows = []
    for rate in review_rates:
        k = max(1, int(np.ceil(len(truth) * rate)))
        selected = order[:k]
        captured = int(truth[selected].sum())
        rows.append(
            {
                "review_rate": rate,
                "alerts": k,
                "frauds_captured": captured,
                "recall": captured / truth.sum(),
                "precision": captured / k,
            }
        )
    return pd.DataFrame(rows)

