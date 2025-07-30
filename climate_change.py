# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:27:16 2025

@author: varra
"""

# Forecasting of Climate Change -(Target Accuracy ~92%)
# Author: Prakash Reddy Varra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =======================
# Step 1: Load & Preprocess Data
# =======================
df = pd.read_csv(r"C:\Users\varra\Downloads\GlobalTemperatures.csv")
df['dt'] = pd.to_datetime(df['dt'])
df = df[['dt', 'LandAverageTemperature']].dropna()

# Feature Engineering
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['quarter'] = df['dt'].dt.quarter

# Use recent data only
df = df[df['year'] >= 1970]

# Features and Target
X = df[['year', 'month', 'quarter']]
y = df['LandAverageTemperature']


# Step 2: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3: Linear Regression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n--- Linear Regression ---")
print("MSE:", mse_lr)
print("R² Score:", r2_lr)
print("Approx Accuracy: {:.2f}%".format(r2_lr * 100))

# Step 4: Random Forest (Controlled Complexity)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest ---")
print("MSE:", mse_rf)
print("R² Score:", r2_rf)
print("Approx Accuracy: {:.2f}%".format(r2_rf * 100))


# Step 5: Cross-Validation

cv_scores = cross_val_score(rf, X, y, scoring='r2', cv=5)
print("\n--- Cross-Validation (Random Forest) ---")
print("Fold R² Scores:", cv_scores)
print("Mean R² Score: {:.4f}".format(cv_scores.mean()))


# Step 6: Residual Plot (Random Forest)

plt.figure(figsize=(8, 4))
plt.hist(y_test - y_pred_rf, bins=50, color='orange', edgecolor='black')
plt.title("Residuals - Random Forest")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# Step 7: Actual vs Predicted Plot

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', linewidth=2)
plt.plot(y_pred_lr[:100], label='Linear Regression', linestyle='--')
plt.plot(y_pred_rf[:100], label='Random Forest', linestyle=':')
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
