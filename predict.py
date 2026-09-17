from pathlib import Path
import argparse
import sys

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import ROOT, make_features

def parse_args():
    parser = argparse.ArgumentParser(description="Predict supermarket transaction sales.")
    parser.add_argument("--unit-price", type=float, default=74.69)
    parser.add_argument("--quantity", type=int, default=7)
    parser.add_argument("--branch", default="A", choices=["A", "B", "C"])
    parser.add_argument("--city", default="Yangon")
    parser.add_argument("--customer-type", default="Member", choices=["Member", "Normal"])
    parser.add_argument("--gender", default="Female", choices=["Female", "Male"])
    parser.add_argument("--product-line", default="Health and beauty")
    parser.add_argument("--payment", default="Ewallet")
    parser.add_argument("--date", default="1/5/2019", help="Date in M/D/YYYY format, e.g. 1/5/2019")
    parser.add_argument("--time", default="13:08", help="Time in HH:MM format, e.g. 13:08")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "linear_regression"])
    return parser.parse_args()


def main():
    args = parse_args()
    row = pd.DataFrame([{
        "Unit price": args.unit_price,
        "Quantity": args.quantity,
        "Branch": args.branch,
        "City": args.city,
        "Customer type": args.customer_type,
        "Gender": args.gender,
        "Product line": args.product_line,
        "Payment": args.payment,
        "Date": args.date,
        "Time": args.time,
    }])
    model_path = ROOT / "models" / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python src/train.py` first.")
    model = joblib.load(model_path)
    prediction = float(model.predict(make_features(row))[0])
    print(f"Model: {args.model}")
    print(f"Predicted total sales: ${prediction:,.2f}")


if __name__ == "__main__":
    main()
