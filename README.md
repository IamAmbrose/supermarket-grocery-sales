# 🛒 Supermart Grocery Sales — Retail Analytics Dataset

This project builds and evaluates machine learning models to **predict supermarket grocery sales** using real-world retail transaction data.  
The goal is to create a **reliable baseline prediction pipeline** for future sales forecasting, inventory planning, and business insights.

👉 [**View Live Dashboard here**](https://supermarket-grocery-sales-48svgo6oj5mlzwxtjjnarc.streamlit.app/)


---


## 📌 **Project Objective**

- Explore historical grocery sales data.
- Engineer relevant features (time, region, category).
- Build multiple machine learning models: Linear Regression, Ridge Regression, Random Forest, and XGBoost.
- Evaluate model performance with clear metrics and visualizations.
- Identify key drivers of sales to inform better decisions.

---

## 📊 **Dataset**

**Columns include:**
- `Order ID`, `Customer Name`
- `Category`, `Sub Category`
- `City`, `State`, `Region`
- `Order Date` (with extracted `Day`, `Month`, `Year`)
- `Sales` (target)
- `Discount`, `Profit`

---

## ⚙️ **Key Steps**

1️ **Data Cleaning:**  
- Checked for missing values, formatted dates, handled categorical values.

2️ **Feature Engineering:**  
- Extracted `Order Day`, `Order Month`, `Order Year` from `Order Date`.
- Encoded categorical columns:
  - `Label Encoding` for tree models.
  - `One-Hot Encoding` for linear models.

3️ **EDA:**  
- Sales trends over time.
- Yearly profit share.
- Correlation heatmaps.
- Scatter plots for predicted vs. actual sales.

4️ **Modeling:**  
- Linear Regression (baseline)
- Ridge Regression (regularized)
- Random Forest Regressor (non-linear)
- XGBoost Regressor (boosted)

5️ **Evaluation:**  
- Metrics: **MSE**, **RMSE**, **MAE**, **R²**
- Visualizations: Actual vs. Predicted scatter plots.

---

## 📈 **Key Findings**

- **Profit** is the strongest single predictor of sales.
- **Discount** alone has limited effect — works best when combined with other features.
- Model capture **~31%** of sales variance (`R² ≈ 0.31`).
- Model handles typical transactions well but struggles with big outliers.

---

## 📊 **Model Results (Best)**

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression| 478.93 | 388.24 | 0.31 |

---

## ✅ **Visuals**

- **Sales over time** shows seasonal patterns.
- **Correlation heatmap** confirms profit–sales link.
- **Predicted vs. Actual plots** reveal good fit for common sales values, weaker fit for large spikes.


---

## 📁 **Project Structure**
```
📦 Supermart Grocery Sales Project
├── data/
│ ├── Supermart Grocery Sales.csv
├── notebooks/
│ ├── EDA.ipynb
│ ├── Modeling.ipynb
├── scripts/
│ ├── data_preprocessing.py
│ ├── train_models.py
├── outputs/
│ ├── predicted_vs_actual.png
│ ├── correlation_heatmap.png
├── README.md
```
---

## 👤 **Author**

- **Ambrose Henry**  
- [LinkedIn](https://www.linkedin.com/in/ambrose-henry-m-30bb84235/)  

---

## 📢 **Questions? Feedback?**

Open an issue or reach out — always happy to improve!

---
