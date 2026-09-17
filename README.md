# Supermarket Sales Prediction

A professional, reproducible machine-learning project that predicts the **Total** value of a supermarket transaction from product, price, quantity, and customer information. The project compares **Linear Regression** with **Random Forest Regressor**, produces evaluation metrics and visualizations, and exposes a reusable prediction script.

## Project objective

Retail teams can use transaction-level sales prediction to estimate expected revenue, compare model behavior, and support planning. The target is the transaction's `Total` amount. The model deliberately excludes columns that are calculated after or directly from the transaction total, such as `Tax 5%`, `cogs`, `gross income`, and the invoice identifier. This prevents target leakage and creates a more defensible prediction workflow.

## Dataset

The sample data contains 1,000 supermarket transactions from three branches in Yangon, Mandalay, and Naypyitaw. It includes product line, unit price, quantity, customer type, gender, branch, city, payment method, date, time, and the observed total. The raw CSV is stored in `data/raw/supermarket_sales.csv` and was sourced from the public repository listed in the references.

## Repository structure

```text
supermarket-sales-prediction/
├── data/raw/                 # Original downloaded CSV
├── data/processed/           # Cleaned modeling data
├── models/                   # Saved trained pipelines
├── reports/figures/          # EDA and model charts
├── reports/metrics/          # JSON evaluation output
├── src/
│   ├── common.py             # Shared loading and feature engineering
│   ├── train.py              # Train, evaluate, save models and charts
│   ├── predict.py            # Predict one or more new transactions
│   └── eda.py                # Exploratory analysis and plots
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/eda.py
python src/train.py
python src/predict.py --unit-price 74.69 --quantity 7 --branch A --city Yangon \
  --customer-type Member --gender Female --product-line "Health and beauty" \
  --payment Ewallet --date 1/5/2019 --time 13:08
```

The training command saves both fitted pipelines in `models/`, a processed dataset in `data/processed/`, charts in `reports/figures/`, and metrics in `reports/metrics/model_metrics.json`.

## Modeling approach

The data is split into training and test sets using an 80/20 split with `random_state=42`. Numeric features are median-imputed and standardized for Linear Regression. Categorical features are most-frequent-imputed and one-hot encoded. Random Forest uses the same transformed feature matrix and is configured with 300 trees, a minimum leaf size of 2, and a fixed random seed for reproducibility.

The models are compared using **MAE**, **RMSE**, and **R²**. MAE is the average absolute error in currency units. RMSE penalizes large errors more strongly. R² measures the proportion of test-set variance explained by the model; higher values are better for all three measures except that lower is better for MAE and RMSE.

## Important limitation

This dataset is historical and small. The prediction is a transaction-level estimate, not a causal forecast of future store revenue. A production deployment should add out-of-time validation, promotions, inventory availability, holidays, competitor prices, and a continuously refreshed data pipeline. If operational data is used, monitor drift and retrain on a schedule.

## References

[1]: https://github.com/selva86/datasets/blob/master/supermarket_sales.csv "Public supermarket sales CSV dataset"
[2]: https://scikit-learn.org/stable/modules/ensemble.html#random-forests "Scikit-learn random forest documentation"
[3]: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares "Scikit-learn linear regression documentation"
