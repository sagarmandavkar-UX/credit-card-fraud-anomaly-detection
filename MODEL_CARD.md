# Model card

## Intended use

Prioritize transactions for analyst review when confirmed fraud labels are scarce. This is a benchmark project, not a deployable payment-blocking system.

## Approach

Transactions are sorted by elapsed time. The earliest 60% is training data, the next 20% is validation, and the last 20% is the future-like test period. Isolation Forest and Local Outlier Factor are fit only on transactions labeled genuine in the training period. Fraud labels are used to compare validation PR-AUC; thresholds are calibrated from unlabeled validation score distributions.

## Result

Local Outlier Factor was selected. A threshold calibrated to flag 0.5% of validation transactions flagged 0.30% of the test period, capturing **39 of 75 frauds (52.0% recall)** with **22.9% precision**. If an analyst instead reviews exactly the top 0.5% of test-period scores, **49 of 75 frauds (65.3%)** are captured at **17.2% precision**.

## Limitations

- The data covers only two days in September 2013 and cannot represent current fraud patterns.
- PCA anonymization prevents meaningful case explanations or investigation workflows.
- Training on known-normal records is semi-supervised and assumes those labels are reliable.
- Alert thresholds drifted between validation and test, demonstrating the need for monitoring and recalibration.
- Accuracy is inappropriate for this 0.172% positive class; PR-AUC, recall, precision, and review volume are emphasized.
- Real deployment requires delayed-label handling, cost-sensitive rules, human review, fairness and privacy assessment, and champion–challenger monitoring.

## Model choice note

An autoencoder is intentionally omitted. The tabular benchmark is already anonymized into PCA components, and adding a neural network would increase operational complexity without guaranteeing a fairer or more interpretable comparison. Isolation Forest and Local Outlier Factor provide distinct tree- and neighborhood-based anomaly baselines.

