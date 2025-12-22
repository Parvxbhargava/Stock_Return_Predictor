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

---

## 🧠 Models Used

### 1. Linear Regression
- Acts as a simple baseline model.
- Assumes a linear relationship between features and next-day returns.
- Trained directly on raw (unscaled) features.
- Coefficients provide interpretability.

### 2. Support Vector Regression (SVR)
- Uses an epsilon-insensitive loss function.
- More robust to noise compared to standard linear regression.
- Requires feature scaling (StandardScaler).
- Both linear and RBF kernels were experimented with; linear SVR performed better for this dataset.

---

## 📊 Features Engineered

The following technical indicators were used as input features:

- **Daily Return (%)** – Day-to-day percentage price change  
- **MA Ratio (MA5 / MA20)** – Short-term vs long-term trend  
- **Volatility** – Rolling standard deviation of returns  
- **Volume Change (%)** – Change in trading volume  
- **High-Low Percentage** – Intraday price range  
- **RSI (Relative Strength Index)** – Momentum indicator  

These features are commonly used in quantitative finance and provide basic signals related to momentum and volatility.

---

## ⚙️ Methodology

- **Time-Series Split**  
  Data is split chronologically (no shuffling) to prevent data leakage.

- **Feature Scaling**  
  - Linear Regression: no scaling required  
  - SVR: features scaled using `StandardScaler`

- **Evaluation Metrics**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Directional Accuracy (up/down prediction)

---

## 📈 Results Summary

| Model | MAE (%) | RMSE (%) | R² | Directional Accuracy |
|------|--------|---------|----|----------------------|
| Baseline (predict 0) | ~1.01 | – | – | – |
| Linear Regression | ~1.02 | ~1.43 | ~0 | ~52% |
| Linear SVR | ~1.03 | ~1.44 | ~0 | ~53% |

**Key observation:**  
Both models perform close to the baseline, which highlights how **noisy and difficult daily stock return prediction is**.  
This behavior is expected and demonstrates the importance of proper evaluation over overfitting or exaggerated claims.

---

## 🔍 Key Learnings

- Daily stock returns contain a high level of noise and low predictability.
- Simple ML models often perform close to baseline on financial time-series data.
- Proper time-series splitting is critical to avoid data leakage.
- Feature scaling is essential for models like SVR.
- Model comparison and honest evaluation are more important than raw accuracy.

How to run :

pip install -r requirements.txt
python src/train_linear_reg.py
python src/svr.py

