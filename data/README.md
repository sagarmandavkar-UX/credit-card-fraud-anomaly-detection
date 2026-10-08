# Data

This project uses the **Credit Card Fraud Detection** dataset published by the Machine Learning Group at Université Libre de Bruxelles (ULB) with Worldline collaborators.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File: `creditcard.csv` (approximately 151 MB)
- Snapshot downloaded: 2026-10-08
- Coverage: 284,807 transactions made by European cardholders over two days in September 2013
- Labels: 492 frauds (0.172%) and 284,315 genuine transactions
- Features: `Time`, `Amount`, 28 anonymized PCA components (`V1`–`V28`), and `Class`
- Kaggle license label: **Database: Open Database, Contents: Database Contents**

The raw CSV is not committed because of its size. `python scripts/download_data.py` downloads the public Kaggle archive without credentials and verifies the expected shape and class count.

The anonymized, two-day sample is useful for methods practice but is not representative of a live payment system. It lacks customer, merchant, device, geography, and delayed-label context.

