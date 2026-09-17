
# 🛒 Supermarket Sales Prediction

A beginner-friendly machine learning project that predicts supermarket transaction sales using product, price, quantity, and customer information.

The project compares two regression algorithms: **Linear Regression** and **Random Forest Regressor** to understand sales prediction and model performance.

---

## 🎯 Project Objective

The main objective of this project is to develop a machine learning model that predicts the **Total sales amount** of a supermarket transaction.

This project helps to:

- Understand supermarket sales data.
- Analyze customer and product information.
- Identify factors related to transaction sales.
- Train machine learning regression models.
- Compare model performance using evaluation metrics.
- Predict sales for new supermarket transactions.

---

## 📋 Project Tasks

### Task 1: Data Collection and Understanding

- Load the supermarket sales dataset.
- Explore rows, columns, and data types.
- Understand product, price, quantity, and customer information.
- Identify the target variable: `Total`.

### Task 2: Data Preprocessing

- Handle missing values.
- Check and remove duplicate records.
- Convert date and time features where needed.
- Select relevant input features.
- Remove unnecessary and target-leaking columns.

### Task 3: Exploratory Data Analysis (EDA)

- Analyze total sales by product line.
- Explore sales by branch and city.
- Understand customer purchasing behavior.
- Visualize quantity, unit price, and total sales.
- Identify useful patterns in the dataset.

### Task 4: Feature Engineering

Prepare the data for machine learning by:

- Selecting numerical features.
- Encoding categorical features.
- Applying preprocessing pipelines.
- Preparing the dataset for model training.

### Task 5: Model Training

Train and compare the following algorithms:

1. **Linear Regression**
2. **Random Forest Regressor**

The dataset is divided into training and testing sets using an 80/20 split.

### Task 6: Model Evaluation

Evaluate both models using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Compare the results to understand how accurately each model predicts sales.

### Task 7: Prediction and Model Saving

- Save trained machine learning pipelines.
- Generate predictions for new supermarket transactions.
- Save evaluation metrics in JSON format.
- Create charts for analysis and reporting.

---

## 📊 Dataset

The dataset contains 1,000 supermarket transactions from three branches in Yangon, Mandalay, and Naypyitaw.

### Main Features

| Feature | Description |
|---|---|
| Product line | Product category |
| Unit price | Price per item |
| Quantity | Number of items purchased |
| Customer type | Member or Normal |
| Gender | Customer gender |
| Branch | Supermarket branch |
| City | Store location |
| Payment | Payment method |
| Date | Transaction date |
| Time | Transaction time |
| Total | Target sales amount |

### Target Variable

```python
Total
```

The model predicts the total value of a supermarket transaction.

---

## 🤖 Algorithms Used

### 1. Linear Regression

A supervised machine learning algorithm used to predict continuous numerical values.

### 2. Random Forest Regressor

An ensemble machine learning algorithm that combines multiple decision trees to predict sales.

---

## 🗂️ Project Structure

```text
supermarket-sales-prediction/
│
├── data/
│   ├── raw/
│   │   └── supermarket_sales.csv
│   └── processed/
│
├── models/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── common.py
│   ├── train.py
│   ├── predict.py
│   └── eda.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
python -m venv .venv
```

### Activate Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac / Linux:**

```bash
source .venv/bin/activate
```

### Install Libraries

```bash
pip install -r requirements.txt
```

### Run EDA

```bash
python src/eda.py
```

### Train Models

```bash
python src/train.py
```

### Make Predictions

```bash
python src/predict.py --unit-price 74.69 --quantity 7 --branch A --city Yangon \
  --customer-type Member --gender Female \
  --product-line "Health and beauty" \
  --payment Ewallet --date 1/5/2019 --time 13:08
```

---

## 📈 Evaluation Metrics

| Metric | Meaning |
|---|---|
| MAE | Average prediction error |
| RMSE | Measures larger prediction errors |
| R² Score | Measures how well the model explains sales variation |

**Lower MAE and RMSE are better. Higher R² is better.**

---

## ⚠️ Project Limitation

This project predicts transaction-level sales, not future monthly or yearly revenue.

The dataset is historical and relatively small. Additional data such as promotions, holidays, inventory, and time-based sales history would be useful for a production forecasting system.

---

## 👩‍💻 Author

**Esha**

Student exploring AI, Data Analytics, and Machine Learning.

### Skills

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Machine Learning` `Regression`

---



⭐ Built with Python and Machine Learning 🚀
