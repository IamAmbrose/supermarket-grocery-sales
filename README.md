# 🛒 Supermart Grocery Sales — Machine Learning Forecasting

This project builds and evaluates machine learning models to **predict supermarket grocery sales** using real-world retail transaction data.  
The goal is to create a **reliable baseline prediction pipeline** for future sales forecasting, inventory planning, and business insights.

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

1️⃣ **Data Cleaning:**  
- Checked for missing values, formatted dates, handled categorical values.

2️⃣ **Feature Engineering:**  
- Extracted `Order Day`, `Order Month`, `Order Year` from `Order Date`.
- Encoded categorical columns:
  - `Label Encoding` for tree models.
  - `One-Hot Encoding` for linear models.

3️⃣ **EDA:**  
- Sales trends over time.
- Yearly profit share.
- Correlation heatmaps.
- Scatter plots for predicted vs. actual sales.

4️⃣ **Modeling:**  
- Linear Regression (baseline)
- Ridge Regression (regularized)
- Random Forest Regressor (non-linear)
- XGBoost Regressor (boosted)

5️⃣ **Evaluation:**  
- Metrics: **MSE**, **RMSE**, **MAE**, **R²**
- Visualizations: Actual vs. Predicted scatter plots.

---

## 📈 **Key Findings**

- **Profit** is the strongest single predictor of sales.
- **Discount** alone has limited effect — works best when combined with other features.
- Models capture **~31–32%** of sales variance (`R² ≈ 0.31–0.32`).
- Random Forest and Ridge Regression perform equally well, indicating limited hidden non-linearity.
- Model handles typical transactions well but struggles with big outliers.

---

## 📊 **Model Results (Best)**

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| Linear Regression | 229,369.89 | 478.93 | 388.24 | 0.31 |
| Ridge Regression | 227,674.81 | 477.15 | 386.05 | 0.32 |
| Random Forest | 226,941.98 | 476.38 | 382.75 | 0.32 |

---

## ✅ **Visuals**

- **Sales over time** shows seasonal patterns.
- **Correlation heatmap** confirms profit–sales link.
- **Predicted vs. Actual plots** reveal good fit for common sales values, weaker fit for large spikes.

---

## 🚀 **Future Improvements**

- Add **holiday flags, day-of-week, rolling averages**.
- Create **interaction features** (e.g., `Discount × Category`).
- Add **customer segments** or loyalty data.
- Bring in **external factors** (weather, local events).
- Tune hyperparameters deeper for RF and XGBoost.
- Try advanced time series or hybrid models for large spikes.

---

## 📁 **Project Structure**

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


---

## 👤 **Author**

- **Your Name**  
- [Your LinkedIn]  
- [Your Email]

---

## ✅ **License**

This project is for **educational purposes** only.  
Dataset source: *[Add source if applicable]*.

---

## 📢 **Questions? Feedback?**

Open an issue or reach out — always happy to improve!

---