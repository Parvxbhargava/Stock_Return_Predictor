import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
ticker = "MSFT"
start_date = "2022-01-01"
end_date = "2025-04-08"
test_frac = 0.2   # temporal test fraction

# -------------------------
# Download data
# -------------------------
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# -------------------------
# Feature engineering
# -------------------------
# daily returns (percent)
data['return'] = data['Close'].pct_change() * 100

# target = next day's return (percent)
data['target'] = data['return'].shift(-1)

# moving averages and ratio
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA_ratio'] = data['MA5'] / data['MA20']

# volatility (rolling std of returns)
data['volatility'] = data['return'].rolling(window=5).std()

# volume change (percent)
data['volume_change'] = data['Volume'].pct_change() * 100

# high-low percent of close
data['hl_pct'] = ((data['High'] - data['Low']) / data['Close']) * 100

# RSI (simple rolling mean version)
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# -------------------------
# Clean up inf / NaN
# -------------------------
# replace infinities and drop NaNs created by pct_change / rolling
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

if len(data) < 50:
    raise RuntimeError("Not enough data after preprocessing. Check the date range or data quality.")

# -------------------------
# Prepare features and target
# -------------------------
feature_cols = ['return', 'MA_ratio', 'volatility', 'volume_change', 'hl_pct', 'RSI']
X = data[feature_cols].copy()
y = data['target'].copy()

# -------------------------
# Temporal train / test split (deterministic)
# -------------------------
split_idx = int(len(X) * (1 - test_frac))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# -------------------------
# Scaling (IMPORTANT: use same scaler for both LR and SVM later)
# -------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------------------------
# Fit Linear Regression (baseline)
# -------------------------
model = LinearRegression()
model.fit(X_train_s, y_train)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test_s)

# -------------------------
# Baseline (naive) predictor: predict 0% change
# -------------------------
baseline_pred = np.zeros_like(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

# -------------------------
# Metrics
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression (scaled features) Results:")
print(f"  Samples: train={len(X_train)}, test={len(X_test)}")
print(f"  MAE:  {mae:.6f} %")
print(f"  RMSE: {rmse:.6f} %")
print(f"  R^2:  {r2:.6f}")
print(f"  Baseline MAE (predict 0): {baseline_mae:.6f} %")

# -------------------------
# Directional accuracy
# -------------------------
# Two ways: (1) include zeros, (2) exclude zero actuals
actual_dir = np.sign(y_test.values)
pred_dir = np.sign(y_pred)
dir_acc_including_zeros = (actual_dir == pred_dir).mean() * 100

mask_nonzero = y_test.values != 0
if mask_nonzero.sum() > 0:
    dir_acc_excluding_zeros = (actual_dir[mask_nonzero] == pred_dir[mask_nonzero]).mean() * 100
else:
    dir_acc_excluding_zeros = np.nan

print(f"  Directional accuracy (including zeros): {dir_acc_including_zeros:.2f}%")
print(f"  Directional accuracy (excluding zero-returns): {dir_acc_excluding_zeros:.2f}%")

# -------------------------
# Last 5 days comparison
# -------------------------
comparison = pd.DataFrame({
    'Actual Return (%)': y_test.tail(5).values,
    'Predicted Return (%)': y_pred[-5:]
}, index=y_test.tail(5).index)
print("\nLast 5 Days Comparison (test set):")
print(comparison)

# -------------------------
# Predict next day's return and price (aligned correctly)
# -------------------------
# The last features row corresponds to the last index in X (which after dropna has target = next day)
last_feat_idx = X.index[-1]
last_features = scaler.transform(X.loc[[last_feat_idx]])  # keep 2D
predicted_return = model.predict(last_features)[0]  # percent

# Use the close corresponding to the SAME feature row (i.e., the reference close)
reference_close = float(data['Close'].loc[last_feat_idx])
predicted_price = reference_close * (1 + predicted_return / 100)

print(f"\nReference date for prediction: {last_feat_idx.date()}")
print(f"Reference close: ${reference_close:.2f}")
print(f"Predicted next day's return: {predicted_return:.4f} %")
print(f"Predicted next day's closing price: ${predicted_price:.2f}")

# -------------------------
# Feature importance (coefficients on scaled features)
# -------------------------
coeffs = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient (scaled)': model.coef_
}).sort_values(by='Coefficient (scaled)', key=abs, ascending=False)
print("\nFeature importance (coefficients on scaled features):")
print(coeffs.to_string(index=False))

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Return', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted Return', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.6)
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.title(f'{ticker} - Actual vs Predicted Daily Returns (Linear Regression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
errors = y_test.values - y_pred
plt.scatter(y_test.index, errors, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Prediction Error (%)')
plt.title(f'{ticker} - Prediction Errors (Actual - Predicted)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
