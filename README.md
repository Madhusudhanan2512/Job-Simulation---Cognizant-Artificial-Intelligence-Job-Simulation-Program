#  Predicting Stock Levels of Products using Machine Learning

This project focuses on predicting the **stock levels of products** using a machine learning-based approach. It involves detailed data preprocessing, exploratory analysis, and predictive modeling using real-world sensor and sales data.

---

##  Project Overview

The goal is to forecast stock levels based on historical sales and sensor data (such as temperature and current stock levels). This can help businesses **automate restocking** and **reduce inventory losses** due to overstocking or understocking.

---

##  Dataset Description

- **sales_data.csv** – Customer transactions with product IDs and timestamps.
- **sensor_stock_levels.csv** – Real-time stock levels captured via sensors.
- **sensor_storage_temperature.csv** – Temperature records from storage units.

Each dataset includes a timestamp, which is aligned during preprocessing for time-series modeling.

---

##  Exploratory Data Analysis (EDA)

Performed using `pandas`, `matplotlib`, and `seaborn`. Key steps included:

- Dropped unnecessary `"Unnamed: 0"` columns.
- Converted `timestamp` columns to `datetime` objects.
- Checked for duplicates and nulls.
- Performed `.info()` to understand schema and memory usage.
- Analyzed `transaction_id` frequency.
- Visualized sales trends and seasonal variations.

Example snippet:

```python
data = pd.read_csv("sample_sales_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.isnull().sum()
data.duplicated().sum()
```

---

##  Model Building

After cleaning the datasets:

1. **Data Merge**:
   - Aligned `sales_data`, `stock_data`, and `temp_data` on `timestamp`.
   - Aggregated stock levels and sales counts.

2. **Feature Engineering**:
   - Created lag features from stock data.
   - Handled missing values via imputation.

3. **Model Training**:
   - Used regression models to predict future stock levels:
     - **Linear Regression**
     - **Decision Tree Regressor**
     - **Random Forest Regressor**

Example snippet:

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
```

---

##  Model Evaluation

Models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

Random Forest showed the most promising results in balancing underfitting and overfitting.

---

##  Key Learnings

- Time-based merging of sensor data is crucial for accuracy.
- Feature lags and temporal trends improve performance.
- Handling missing sensor data smartly impacts prediction quality.

---

##  Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Jupyter Notebook
