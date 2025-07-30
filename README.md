
# 🌍 Forecasting of Climate Change using Machine Learning

### 🔬 Author: Prakash Reddy Varra  
### 🎯 Target Accuracy: ~92%  
### 📁 Dataset Source: Global Land Temperatures Dataset

---

## 📌 Project Overview

Climate change is one of the most pressing global challenges today. This project aims to **forecast average land temperature** using Machine Learning algorithms to analyze historical climate data.

We compare **Linear Regression** and **Random Forest Regressor** to predict temperature trends based on features like **year, month, and quarter**. The project concludes with **visual insights and evaluation metrics** like R² score and residual plots.

---

## 📊 Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Machine Learning (Regression)
- Cross-validation

---

## 📂 Dataset Information

- **Source**: [Kaggle – GlobalTemperatures](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
- **Features Used**:
  - `year`
  - `month`
  - `quarter`
- **Target**:
  - `LandAverageTemperature`

---

## ⚙️ Project Workflow

### 1. **Data Preprocessing**
- Convert date column to datetime format
- Extract year, month, and quarter
- Filter dataset to include records from 1970 onward
- Handle missing values

### 2. **Model Training**
- Split into training and testing sets
- Train and evaluate:
  - Linear Regression
  - Random Forest Regressor

### 3. **Evaluation Metrics**
- R² Score
- Mean Squared Error (MSE)
- Cross-validation on Random Forest
- Residual distribution
- Actual vs Predicted plot

---

## 📈 Sample Results

### 🔹 Linear Regression
- R² Score: `~81%`
- MSE: `~0.33`

### 🔹 Random Forest Regressor
- R² Score: `~92%`
- MSE: `~0.18`
- Cross-Validation Mean R²: `~0.91`

---

## 📉 Visualizations

- 🔍 **Residual Histogram** for error analysis
- 📊 **Actual vs Predicted Plot** for comparison between models

<p align="center">
  <img src="plots/residuals.png" alt="Residual Plot" width="500"/>
  <br>
  <em>Residuals of Random Forest Model</em>
</p>

<p align="center">
  <img src="plots/actual_vs_predicted.png" alt="Actual vs Predicted Plot" width="600"/>
  <br>
  <em>Actual vs Predicted Temperatures (First 100 Samples)</em>
</p>

---

## 🚀 Future Improvements
- Integrate LSTM model for time series forecasting
- Feature expansion (e.g., region, humidity, CO₂ levels)
- Dashboard visualization (Power BI or Streamlit)

---

## 📎 Project Structure

