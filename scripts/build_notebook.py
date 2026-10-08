from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/fraud_anomaly_detection.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Credit Card Fraud Anomaly Detection\n\n## tl;dr\n\n"
            "Local Outlier Factor produced **0.176 test PR-AUC** and **0.913 ROC-AUC**. A threshold calibrated to flag "
            "0.5% of validation transactions flagged 0.30% of the future test period and captured **52.0% of frauds** at "
            "**22.9% precision**. Reviewing exactly the top 0.5% of test scores captures **65.3% of frauds**."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The goal is to rank unusual transactions for limited human review. Isolation Forest and Local Outlier Factor are fit only on known-normal rows from the earliest 60% of transactions. The next 20% selects the model and calibrates a score threshold; the latest 20% is untouched until final evaluation.\n\n"
            "### Key Assumptions\n\n"
            "- `Time` order approximates deployment order within the two-day sample.\n"
            "- Known genuine training rows are suitable examples of normal activity.\n"
            "- The 0.5% validation alert-rate target is illustrative, not an actual bank capacity."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, os, subprocess, sys\nimport pandas as pd\nfrom IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src')\n"
            "if not (ROOT / 'data/raw/creditcard.csv').exists():\n"
            "    subprocess.run([sys.executable, str(ROOT / 'scripts/download_data.py')], cwd=ROOT, check=True)\n"
            "subprocess.run([sys.executable, str(ROOT / 'scripts/run_analysis.py')], cwd=ROOT, env=env, check=True, capture_output=True)\n"
            "summary = json.loads((ROOT / 'reports/summary.json').read_text())\nsummary"
        ),
        nbf.v4.new_markdown_cell("## Data\n\nThe ULB/Worldline benchmark contains 284,807 transactions and only 492 frauds. The extreme imbalance makes ordinary accuracy uninformative."),
        nbf.v4.new_code_cell(
            "data = pd.read_csv(ROOT / 'data/raw/creditcard.csv')\n"
            "pd.DataFrame({'transactions': [len(data)], 'frauds': [int(data.Class.sum())], 'fraud_rate': [data.Class.mean()], 'missing_cells': [int(data.isna().sum().sum())]})"
        ),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/class_imbalance.png', width=750))"),
        nbf.v4.new_markdown_cell("## Results\n\n### Ranking performance\n\nPR-AUC is emphasized because it reflects the precision–recall tradeoff on the rare fraud class. The test no-skill baseline is its fraud prevalence."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/precision_recall_curves.png', width=800))\npd.read_csv(ROOT / 'reports/test_metrics.csv').round(4)"),
        nbf.v4.new_markdown_cell("### Validation-calibrated operating point\n\nThe score distribution shifted: thresholds intended to flag 0.5% of validation transactions flagged only 0.30–0.34% of test transactions. This is a monitoring signal, not an implementation detail to hide."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/operating_point_comparison.png', width=800))\npd.read_csv(ROOT / 'reports/validation_metrics.csv').round(4)"),
        nbf.v4.new_markdown_cell("### Analyst review capacity\n\nThe curve ranks the future test batch and asks what fraction of confirmed fraud would be captured at fixed review capacities."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/review_capacity_curve.png', width=800))\npd.read_csv(ROOT / 'reports/review_capacity_curve.csv').assign(review_rate=lambda d: d.review_rate * 100, recall=lambda d: d.recall * 100, precision=lambda d: d.precision * 100).round(2)"),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- Use anomaly scores to prioritize review, not automatically block transactions.\n"
            "- At an exact 0.5% batch review capacity, Local Outlier Factor captures 49 of 75 future-period frauds.\n"
            "- Threshold transport changed alert volume, so production monitoring must track score drift and review load.\n"
            "- Real deployment needs recent features, delayed labels, cost-sensitive decisions, explanations, and analyst feedback.\n"
            "- Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)."
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

