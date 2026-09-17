from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from common import ROOT, load_data

FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")


def print_bordered_table(df):
    columns = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    all_rows = [columns] + rows
    widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(columns))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(border)
    print("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(columns)) + " |")
    print(border)
    for row in rows:
        print("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |")
    print(border)


def main():
    df = load_data()
    df["Date"] = __import__("pandas").to_datetime(df["Date"])

    plt.figure(figsize=(9, 5))
    sns.histplot(df["Total"], bins=30, kde=True, color="#2563eb")
    plt.title("Distribution of Transaction Sales")
    plt.xlabel("Total sales value")
    plt.tight_layout()
    plt.savefig(FIGURES / "sales_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    category_sales = df.groupby("Product line", as_index=False)["Total"].mean().sort_values("Total", ascending=False)
    sns.barplot(data=category_sales, x="Total", y="Product line", color="#0f766e")
    plt.title("Average Transaction Sales by Product Line")
    plt.xlabel("Average total sales value")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "average_sales_by_product_line.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Unit price", y="Total", size="Quantity", hue="Customer type", alpha=0.75)
    plt.title("Price, Quantity, and Transaction Sales")
    plt.tight_layout()
    plt.savefig(FIGURES / "price_quantity_sales.png", dpi=180)
    plt.close()

    summary = df[["Unit price", "Quantity", "Tax 5%", "Total", "Rating"]].describe().round(2)
    summary.to_csv(ROOT / "reports" / "eda_summary.csv")
    print(f"Saved EDA charts to {FIGURES}")
    print()
    print_bordered_table(summary)


if __name__ == "__main__":
    main()
