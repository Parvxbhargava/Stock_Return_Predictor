# Stock Return Prediction using Linear Regression and SVR

This project explores the use of basic machine learning models to predict **daily stock returns** using historical price data.  
It compares **Linear Regression** and **Support Vector Regression (SVR)** on the same dataset to understand their behavior on noisy financial time-series data.

The goal of this project is **not** to beat the stock market, but to demonstrate a clean and correct ML workflow, proper time-series handling, and fair model comparison.

---

## 📌 Project Overview

- Historical stock data for **Microsoft (MSFT)** is collected using the `yfinance` library.
- Several commonly used **technical indicators** are engineered as features.
- Two models are implemented:
  - Linear Regression (baseline)
  - Support Vector Regression (SVR)
- Both models are evaluated using multiple metrics, including directional accuracy.
- A next-day return and price prediction is generated using the trained models.


How to run :

pip install -r requirements.txt
python train_linear_reg.py
python svm.py

