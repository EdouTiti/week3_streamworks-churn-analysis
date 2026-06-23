# StreamWorks — Customer Churn Analysis

Predictive analytics project for StreamWorks Media investigating customer churn. The analysis includes data cleaning, feature engineering, statistical testing, model training (logistic and linear regression), and visualizations to support retention strategies.

## Table of Contents
- Project overview
- Dataset
- Requirements
- How to run
- Key findings
- Files
- Author

## Project overview

This repository contains a reproducible analysis and modelling pipeline to explore drivers of customer churn and predict churn probability using Python and scikit-learn. The goal is to provide actionable insights for customer retention.

## Dataset

- Source: StreamWorks user data (included in `data/`).
- Cleaned / transformed: `streamworks_user_data_transformed.csv` (used for modelling).

## Requirements

Install dependencies into a virtual environment:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run

- Open the project notebook for an interactive walkthrough: [Week3project_Edou Ndong.ipynb](Week3project_Edou%20Ndong.ipynb)
- Or run your preferred Python scripts after activating the environment above.

## Key findings (summary)

- Identified features correlated with churn (see correlation heatmap).
- Performed chi-square and t-tests for group differences.
- Built a logistic regression model for churn classification; performance visualized with an ROC curve.
- Built a linear regression model for a continuous target and inspected residuals.

## Analysis & Interpretation

- Engagement and usage features (e.g., total watch hours, average session length, recency of activity) show the strongest negative correlation with retention: lower engagement associates with higher churn probability.
- Demographic and subscription features (tenure, plan type) provide additional predictive signal when combined with behavioral data.
- Statistical tests indicate significant differences between churners and non-churners on several numeric features; categorical tests (chi-square) highlight segments with disproportionate churn rates.
- The logistic regression model offers a transparent baseline for classification; inspect coefficients to interpret feature impact and monitor for multicollinearity.

## Business Questions & Answers

- Q: Which customer segments are most at risk of churn?
	- A: Low-engagement users (low watch hours, few active days), new users with short tenure, and users on lower-tier plans show elevated churn risk.
- Q: What are the top predictors of churn?
	- A: Low recent activity, sudden drops in usage, fewer sessions per week, and low feature adoption are consistent top predictors in the feature importance analysis.
- Q: How reliable is the model for targeting retention campaigns?
	- A: Use model probabilities as a ranking/prioritization tool rather than a hard decision. Validate on hold-out data and A/B test campaigns to measure lift before full deployment.

## Recommendations

- Prioritized retention campaigns: target high-risk users (top percentile by predicted churn probability) with personalized offers and content recommendations.
- Improve onboarding and early engagement: implement in-app guides, tailored recommendations, and nudges within the first 30 days to reduce early churn.
- Monitor and retrain: establish a monthly retraining cadence and track key metrics (AUC, precision@k, calibration) to detect model drift.
- Experiment and measure: run randomized controlled trials (A/B tests) for interventions to measure true causal impact on churn.
- Data and privacy: anonymize PII, use aggregated features where possible, and comply with regional privacy regulations before using user-level predictions in campaigns.

## Risks & Mitigations

- False positives: targeting users incorrectly can waste marketing budget and annoy customers. Mitigation: prefer probability thresholds tuned for precision, and A/B test interventions.
- Data bias: training data may reflect historical biases (e.g., certain demographics underrepresented). Mitigation: assess fairness metrics and, if necessary, collect additional representative data or apply bias-mitigation techniques.
- Model drift: user behavior and product changes reduce model performance over time. Mitigation: monitor model performance, retrain regularly, and deploy an incremental validation pipeline.
- Privacy and compliance: using behavioral data for intervention can raise legal concerns. Mitigation: consult legal/privacy teams and implement opt-outs for users.

## Reproduce model training (example commands)

1) Prepare environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Execute the notebook end-to-end (non-interactive) to reproduce the analysis and artifacts:

```bash
# Option A: run with nbconvert
jupyter nbconvert --to notebook --execute "Week3project_Edou Ndong.ipynb" --output executed_notebook.ipynb

# Option B: run with papermill (recommended for parameterization)
pip install papermill
papermill "Week3project_Edou Ndong.ipynb" output.ipynb
```

3) If you have a training script (recommended for CI / reproducibility), example:

```bash
python scripts/train_models.py \
	--data data/streamworks_user_data_transformed.csv \
	--output models/ \
	--random-seed 42
```

4) Save model artifacts and metrics (example within scripts/notebooks):

```python
from joblib import dump
dump(clf, 'models/logistic_model.pkl')
# and save evaluation metrics to JSON/CSV for tracking
```

Notes:
- The repository currently includes the analysis notebook; for production-ready reproducibility, consider adding a `scripts/train_models.py` and a small `Makefile` or `tasks` scripts to standardize steps.

## Files

- `Week3project_Edou Ndong.ipynb` — analysis notebook and modelling pipeline
- `streamworks_user_data_transformed.csv` — processed dataset used in the notebook
- `Week3_Report.pdf` — written report with business recommendations
- `requirements.txt` — Python dependencies
- `data/` — raw and intermediate CSVs
- `images/` — figures produced during analysis (ROC, residuals, heatmap, etc.)

## Author

Edou Ndong — Data Analytics & Data Engineering

---
