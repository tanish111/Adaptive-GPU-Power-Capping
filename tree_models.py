import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import os

# Create the models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Load the dataset
data = pd.read_csv('/Users/tanishdesai37/Downloads/TempData/merged_results.csv')

# Define features and target variable
X = data[['Temperature (C)', 'Memory Utilization (%)', 'Frequency','GPU Utilization (%)']]
y = data['PowerCap']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to evaluate model using cross-validation
def evaluate_model(model, X_train, y_train):
    mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    mae_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
    r2_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kf)
    return -mse_scores.mean(), -mae_scores.mean(), r2_scores.mean()

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
joblib.dump(linear_regressor, './models/linear_regressor.pkl')
mse_linear, mae_linear, r2_linear = evaluate_model(linear_regressor, X_train, y_train)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)
joblib.dump(rf_regressor, './models/rf_regressor.pkl')
mse_rf, mae_rf, r2_rf = evaluate_model(rf_regressor, X_train, y_train)

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
joblib.dump(dt_regressor, './models/dt_regressor.pkl')
mse_dt, mae_dt, r2_dt = evaluate_model(dt_regressor, X_train, y_train)

# XGBoost Regressor
xgb_regressor = XGBRegressor(random_state=42)
xgb_regressor.fit(X_train, y_train)
joblib.dump(xgb_regressor, './models/xgb_regressor.pkl')
mse_xgb, mae_xgb, r2_xgb = evaluate_model(xgb_regressor, X_train, y_train)

# CatBoost Regressor
catboost_regressor = CatBoostRegressor(random_state=42, verbose=0)
catboost_regressor.fit(X_train, y_train)
joblib.dump(catboost_regressor, './models/catboost_regressor.pkl')
mse_catboost, mae_catboost, r2_catboost = evaluate_model(catboost_regressor, X_train, y_train)

# Print mean squared errors, mean absolute errors, and R-squared values
print(f"Linear Regression MSE: {mse_linear}, MAE: {mae_linear}, R²: {r2_linear}")
print(f"Random Forest Regressor MSE: {mse_rf}, MAE: {mae_rf}, R²: {r2_rf}")
print(f"Decision Tree Regressor MSE: {mse_dt}, MAE: {mae_dt}, R²: {r2_dt}")
print(f"XGBoost Regressor MSE: {mse_xgb}, MAE: {mae_xgb}, R²: {r2_xgb}")
print(f"CatBoost Regressor MSE: {mse_catboost}, MAE: {mae_catboost}, R²: {r2_catboost}")

# Save the results to a CSV file
results = {
    'Model': ['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor', 'XGBoost Regressor', 'CatBoost Regressor'],
    'MSE': [mse_linear, mse_rf, mse_dt, mse_xgb, mse_catboost],
    'MAE': [mae_linear, mae_rf, mae_dt, mae_xgb, mae_catboost],
    'R²': [r2_linear, r2_rf, r2_dt, r2_xgb, r2_catboost]
}
results_df = pd.DataFrame(results)
results_df.to_csv('./models/result_test.csv', index=False)

# Load the new dataset for testing
new_data = pd.read_csv('/Users/tanishdesai37/Downloads/TempData/feature_data_single_0.csv')

# Filter rows where PowerCap is 90
filtered_data = new_data[new_data['PowerCap'] == 110000]

# Define features for prediction
X_new = filtered_data[['Temperature (C)', 'Memory Utilization (%)', 'Frequency', 'GPU Utilization (%)']]

# Predict PowerCap using the trained models
y_pred_new_linear = linear_regressor.predict(X_new)
y_pred_new_rf = rf_regressor.predict(X_new)
y_pred_new_dt = dt_regressor.predict(X_new)
y_pred_new_xgb = xgb_regressor.predict(X_new)
y_pred_new_catboost = catboost_regressor.predict(X_new)

# Calculate the average predicted PowerCap for each model
avg_pred_linear = y_pred_new_linear.mean()
avg_pred_rf = y_pred_new_rf.mean()
avg_pred_dt = y_pred_new_dt.mean()
avg_pred_xgb = y_pred_new_xgb.mean()
avg_pred_catboost = y_pred_new_catboost.mean()

# Print the average predicted PowerCap for each model
print(f"Average predicted PowerCap (Linear Regression): {avg_pred_linear}")
print(f"Average predicted PowerCap (Random Forest Regressor): {avg_pred_rf}")
print(f"Average predicted PowerCap (Decision Tree Regressor): {avg_pred_dt}")
print(f"Average predicted PowerCap (XGBoost Regressor): {avg_pred_xgb}")
print(f"Average predicted PowerCap (CatBoost Regressor): {avg_pred_catboost}")
