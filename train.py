from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from common import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, ROOT, load_data, split_xy

MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
for directory in (MODELS, PROCESSED, METRICS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_models():
    return {
        "linear_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ]),
    }


def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }, predictions


def main():
    df = load_data()
    x, y = split_xy(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    x.to_csv(PROCESSED / "model_features.csv", index=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    metrics = {}
    predictions_for_plot = {}
    for name, model in build_models().items():
        model.fit(train_x, train_y)
        model_metrics, predictions = evaluate(model, test_x, test_y)
        metrics[name] = model_metrics
        predictions_for_plot[name] = predictions
        joblib.dump(model, MODELS / f"{name}.joblib")

    with open(METRICS / "model_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"target": "Total", "test_size": 0.2, "random_state": 42, "models": metrics}, handle, indent=2)

    comparison = pd.DataFrame({"actual": test_y.to_numpy(), **predictions_for_plot})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=comparison, x="actual", y="random_forest", color="#0f766e", label="Random Forest", alpha=0.75)
    sns.scatterplot(data=comparison, x="actual", y="linear_regression", color="#2563eb", label="Linear Regression", alpha=0.55)
    limits = [0, max(comparison.max()) * 1.05]
    plt.plot(limits, limits, "--", color="black", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual total")
    plt.ylabel("Predicted total")
    plt.title("Actual vs Predicted Supermarket Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved trained pipelines to {MODELS}")


if __name__ == "__main__":
    main()
