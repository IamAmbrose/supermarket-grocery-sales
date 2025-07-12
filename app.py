# streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io

# Load data & model
data = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Load your trained Linear Regression & scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


# Dashboard title
st.title("🛒 Supermart Grocery Sales — Interactive Dashboard")

st.markdown("""
Welcome!  
Explore the data, see trends, check correlations — and predict new sales!
""")

# Key metrics
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

total_sales = data['Sales'].sum()
total_profit = data['Profit'].sum()
avg_discount = data['Discount'].mean()

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
col3.metric("🏷️ Average Discount", f"{avg_discount:.2%}")

# Sales over time
st.subheader("Sales Over Time")

data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
sales_over_time = data.groupby(data['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
sales_over_time['Order Date'] = sales_over_time['Order Date'].dt.to_timestamp()

st.line_chart(sales_over_time.rename(columns={'Order Date': 'index'}).set_index('index'))

# Correlation heatmap
st.subheader("Correlation Heatmap")

numeric_cols = ['Sales', 'Profit', 'Discount']
corr = data[numeric_cols].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Actual vs Predicted scatter plot (static from your model)
st.subheader("Predicted vs Actual Sales (Test Data)")

# For demonstration, simulate test split
features = data.drop(columns=['Order ID', 'Customer Name', 'Order Date', 'Sales'])
target = data['Sales']

# Simple split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)

y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)

fig2, ax2 = plt.subplots()
sns.scatterplot(x='Actual', y='Predicted', data=results, ax=ax2)
ax2.plot([results['Actual'].min(), results['Actual'].max()],
         [results['Actual'].min(), results['Actual'].max()], 'r--')
ax2.set_title("Actual vs Predicted Sales")
st.pyplot(fig2)

# Make a new prediction!
st.subheader("🔍 Predict New Sales")

category = st.selectbox("Category", ["Furniture", "Technology", "Office Supplies"])
region = st.selectbox("Region", ["East", "West", "Central", "South"])
discount = st.slider("Discount", 0.0, 1.0, 0.1)
profit = st.number_input("Profit", min_value=0.0, value=100.0)
order_day = st.slider("Order Day", 1, 31, 15)
order_month = st.slider("Order Month", 1, 12, 6)
order_year = st.selectbox("Order Year", [2015, 2016, 2017, 2018])

# Simulate dummy encoding
X_input = pd.DataFrame({
    'Discount': [discount],
    'Profit': [profit],
    'Order Day': [order_day],
    'Order Month': [order_month],
    'Order Year': [order_year],
    'Category_Furniture': [1 if category == "Furniture" else 0],
    'Category_Technology': [1 if category == "Technology" else 0],
    'Region_East': [1 if region == "East" else 0],
    'Region_West': [1 if region == "West" else 0],
})

# Scale
X_input_scaled = scaler.transform(X_input)

# Predict
if st.button("Predict Sales"):
    y_pred_log = model.predict(X_input_scaled)
    y_pred = np.expm1(y_pred_log)
    st.success(f" **Predicted Sales: ${y_pred[0]:,.2f}**")

