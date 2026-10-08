from pathlib import Path

import numpy as np

from fraud_detection.modeling import alert_threshold, evaluate_scores, load_transactions, review_curve, time_ordered_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/creditcard.csv"


def test_dataset_contract() -> None:
    data = load_transactions(DATA)
    assert data.shape == (284807, 31)
    assert int(data["Class"].sum()) == 492
    assert data["Time"].is_monotonic_increasing
    assert not data.isna().any().any()


def test_time_ordered_split_is_non_overlapping() -> None:
    data = load_transactions(DATA)
    train, validation, test = time_ordered_split(data)
    assert len(train) + len(validation) + len(test) == len(data)
    assert train["Time"].max() <= validation["Time"].min()
    assert validation["Time"].max() <= test["Time"].min()


def test_operating_metrics() -> None:
    truth = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    threshold = alert_threshold(scores, 0.4)
    metrics = evaluate_scores(truth, scores, threshold)
    curve = review_curve(truth, scores, [0.4])
    assert metrics["frauds_captured"] == 2
    assert metrics["precision"] == 1.0
    assert curve.iloc[0]["recall"] == 1.0

