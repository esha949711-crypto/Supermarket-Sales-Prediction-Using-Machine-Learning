from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "supermarket_sales.csv"

TARGET = "Total"
NUMERIC_FEATURES = ["Unit price", "Quantity", "hour", "day", "month", "weekday"]
CATEGORICAL_FEATURES = [
    "Branch", "City", "Customer type", "Gender", "Product line", "Payment"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LEAKAGE_COLUMNS = ["Tax 5%", "cogs", "gross margin percentage", "gross income"]


def load_data(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load and validate the raw transaction dataset."""
    df = pd.read_csv(path)
    raw_features = ["Unit price", "Quantity"] + CATEGORICAL_FEATURES
    required = set(raw_features + [TARGET, "Date", "Time"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features without using post-sale accounting fields."""
    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    result["Time"] = pd.to_datetime(result["Time"], format="%H:%M", errors="coerce")
    result["day"] = result["Date"].dt.day
    result["month"] = result["Date"].dt.month
    result["weekday"] = result["Date"].dt.dayofweek
    result["hour"] = result["Time"].dt.hour
    if result[FEATURES].isna().any().any():
        bad = result[FEATURES].isna().sum()
        raise ValueError(f"Unparseable feature values found: {bad[bad > 0].to_dict()}")
    return result[FEATURES]


def split_xy(df: pd.DataFrame):
    """Return feature matrix and target vector."""
    return make_features(df), df[TARGET].astype(float)


def clean_cli_categories(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values]
