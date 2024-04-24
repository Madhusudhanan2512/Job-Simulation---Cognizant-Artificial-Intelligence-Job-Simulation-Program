import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# Loading the data sets
sales_data = pd.read_csv("E:\sales.csv")
stock_data = pd.read_csv("E:\sensor_stock_levels.csv")
temp_data = pd.read_csv("E:\sensor_storage_temperature.csv")

# Creating target and predictor variable
x = merged_data.drop('estimated_stock_pct',axis=1)
y = merged_data['estimated_stock_pct']

# Training the model
scaled_X = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                                    test_size = 0.3, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_reg.fit(X_train, y_train)

# Make predictions on the train data and test data
predictions = rf_reg.predict(X_test)
predict_tr = rf_reg.predict(X_train)

# Finish by compputing the errors
mse = mean_squared_error(y_train, predict_tr)
print(f'Mean Squared Error on Test Data: {round(mse, 2)}')

mae = mean_absolute_error(y_train, predict_tr)
print(f'Mean Absolute Error on Test Data: {round(mae, 2)}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error on Test Data: {round(rmse, 2)}')