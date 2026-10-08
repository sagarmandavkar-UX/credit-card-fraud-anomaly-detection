from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from fraud_detection.modeling import FEATURE_COLUMNS, alert_threshold, evaluate_scores, load_transactions, review_curve, time_ordered_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/creditcard.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ALERT_RATE = 0.005
BLUE, GOLD, PURPLE, ORANGE = "#2463A9", "#C58B24", "#8A5A9B", "#D26A3A"


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def score_in_batches(model, values: np.ndarray, batch_size: int = 10_000) -> np.ndarray:
    return np.concatenate([-model.decision_function(values[start : start + batch_size]) for start in range(0, len(values), batch_size)])


def main(quick: bool = False) -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    data = load_transactions(DATA)
    train, validation, test = time_ordered_split(data)
    normal_train = train[train["Class"].eq(0)]
    rng = np.random.default_rng(42)
    scaler = RobustScaler().fit(normal_train[FEATURE_COLUMNS])
    validation_x = scaler.transform(validation[FEATURE_COLUMNS])
    test_x = scaler.transform(test[FEATURE_COLUMNS])

    if_train_n = min(40_000 if quick else 100_000, len(normal_train))
    if_indices = rng.choice(len(normal_train), size=if_train_n, replace=False)
    if_train_x = scaler.transform(normal_train.iloc[if_indices][FEATURE_COLUMNS])
    isolation_forest = IsolationForest(
        n_estimators=150 if quick else 350,
        max_samples=min(4096, if_train_n),
        contamination="auto",
        n_jobs=-1,
        random_state=42,
    ).fit(if_train_x)

    lof_train_n = min(5_000 if quick else 12_000, len(normal_train))
    lof_indices = rng.choice(len(normal_train), size=lof_train_n, replace=False)
    lof_train_x = scaler.transform(normal_train.iloc[lof_indices][FEATURE_COLUMNS])
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1).fit(lof_train_x)

    models = {"Isolation Forest": isolation_forest, "Local Outlier Factor": lof}
    validation_scores = {name: score_in_batches(model, validation_x) for name, model in models.items()}
    validation_metrics = []
    thresholds = {}
    for name, scores in validation_scores.items():
        thresholds[name] = alert_threshold(scores, ALERT_RATE)
        row = evaluate_scores(validation["Class"].to_numpy(), scores, thresholds[name])
        row["model"] = name
        validation_metrics.append(row)
    validation_table = pd.DataFrame(validation_metrics).sort_values("pr_auc", ascending=False)
    selected_model = str(validation_table.iloc[0]["model"])

    test_scores = {name: score_in_batches(model, test_x) for name, model in models.items()}
    test_metrics = []
    for name, scores in test_scores.items():
        row = evaluate_scores(test["Class"].to_numpy(), scores, thresholds[name])
        row["model"] = name
        row["threshold_from_validation"] = thresholds[name]
        test_metrics.append(row)
    test_table = pd.DataFrame(test_metrics).sort_values("pr_auc", ascending=False)
    selected_metrics = test_table[test_table["model"].eq(selected_model)].iloc[0].to_dict()

    review_rates = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    review = review_curve(test["Class"].to_numpy(), test_scores[selected_model], review_rates)
    review.to_csv(REPORTS / "review_capacity_curve.csv", index=False)
    validation_table.to_csv(REPORTS / "validation_metrics.csv", index=False)
    test_table.to_csv(REPORTS / "test_metrics.csv", index=False)
    pd.DataFrame({"time": test["Time"].to_numpy(), "actual_fraud": test["Class"].to_numpy(), **{name.lower().replace(" ", "_") + "_score": scores for name, scores in test_scores.items()}}).to_csv(REPORTS / "test_anomaly_scores.csv", index=False)

    summary = {
        "rows": int(len(data)),
        "frauds": int(data["Class"].sum()),
        "fraud_rate": float(data["Class"].mean()),
        "split": {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
        "training_strategy": "Models fit only on known-normal transactions from the earliest 60% of time-ordered data.",
        "alert_rate_target": ALERT_RATE,
        "selected_model": selected_model,
        "selection_metric": "validation PR-AUC",
        "test_pr_auc": float(selected_metrics["pr_auc"]),
        "test_roc_auc": float(selected_metrics["roc_auc"]),
        "test_alerts": int(selected_metrics["alerts"]),
        "test_actual_alert_rate": float(selected_metrics["alert_rate"]),
        "test_frauds": int(selected_metrics["frauds"]),
        "test_frauds_captured": int(selected_metrics["frauds_captured"]),
        "test_precision_at_alert_rate": float(selected_metrics["precision"]),
        "test_recall_at_alert_rate": float(selected_metrics["recall"]),
        "test_false_positives": int(selected_metrics["false_positives"]),
        "caveat": "This anonymized two-day 2013 sample is a methods benchmark, not production fraud evidence.",
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    counts = data["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(["Genuine", "Fraud"], counts.values, color=[BLUE, ORANGE])
    ax.set_yscale("log")
    ax.set(title="Extreme class imbalance in the transaction dataset", ylabel="Transactions (log scale)", xlabel="")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.2, f"{value:,}", ha="center")
    save_figure(FIGURES / "class_imbalance.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, scores in test_scores.items():
        precision, recall, _ = precision_recall_curve(test["Class"], scores)
        ax.plot(recall, precision, lw=2.2, color=BLUE if name == "Isolation Forest" else GOLD, label=f"{name} (AP={average_precision_score(test['Class'], scores):.3f})")
    ax.axhline(test["Class"].mean(), color="#666666", linestyle="--", label=f"No-skill ({test['Class'].mean():.4f})")
    ax.set(title="Fraud precision–recall curves on the future test period", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    save_figure(FIGURES / "precision_recall_curves.png")

    plot_metrics = test_table.set_index("model")[["precision", "recall"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(plot_metrics))
    width = 0.35
    ax.bar(x - width / 2, plot_metrics["precision"] * 100, width, color=GOLD, label="Precision")
    ax.bar(x + width / 2, plot_metrics["recall"] * 100, width, color=BLUE, label="Recall")
    ax.set_xticks(x, plot_metrics.index)
    ax.set(title=f"Performance using a validation-calibrated {ALERT_RATE:.1%} alert threshold", ylabel="Percent", xlabel="", ylim=(0, 100))
    ax.legend(frameon=False)
    save_figure(FIGURES / "operating_point_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(review["review_rate"] * 100, review["recall"] * 100, marker="o", lw=2.2, color=BLUE)
    ax.plot([0.1, 5], [0.1, 5], linestyle="--", color="#777777", label="Random review")
    ax.set(title=f"Fraud captured as {selected_model} review capacity expands", xlabel="Transactions sent to review (%)", ylabel="Frauds captured (%)", xlim=(0, 5.1), ylim=(0, 100))
    ax.legend(frameon=False)
    save_figure(FIGURES / "review_capacity_curve.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
