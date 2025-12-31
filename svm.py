import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Config 
# -------------------------
ticker = "MSFT"
start_date = "2022-01-01"
end_date = "2025-04-08"
test_frac = 0.2
feature_cols = ['return', 'MA_ratio', 'volatility', 'volume_change', 'hl_pct', 'RSI']

# -------------------------
# Download data
# -------------------------
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# -------------------------
# Feature engineering 
# -------------------------
data['return'] = data['Close'].pct_change() * 100
data['target'] = data['return'].shift(-1)

data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA_ratio'] = data['MA5'] / data['MA20']

data['volatility'] = data['return'].rolling(window=5).std()
data['volume_change'] = data['Volume'].pct_change() * 100
data['hl_pct'] = ((data['High'] - data['Low']) / data['Close']) * 100

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# -------------------------
# Clean up
# -------------------------
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
if len(data) < 50:
    raise RuntimeError("Not enough data after preprocessing.")

# -------------------------
# Prepare X, y
# -------------------------
X = data[feature_cols].copy()
y = data['target'].copy()

# -------------------------
# Deterministic temporal split 
# -------------------------
split_idx = int(len(X) * (1 - test_frac))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Samples: train={len(X_train)}, test={len(X_test)}")

# -------------------------
# Scale only for SVR 
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'C': [1, 10, 50, 100],             # start smaller; expand if needed
    'epsilon': [0.01, 0.05, 0.1, 0.2],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.05],  # only for rbf/poly
}

grid = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0
)

print("Running GridSearchCV (time-series CV)...")
grid.fit(X_train_scaled, y_train)
svr = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# -------------------------
# Predict
# -------------------------
y_pred = svr.predict(X_test_scaled)

# -------------------------
# Metrics
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nSVR Results:")
print(f"  MAE:  {mae:.6f} %")
print(f"  RMSE: {rmse:.6f} %")
print(f"  R^2:  {r2:.6f}")

# Directional accuracy
actual_dir = np.sign(y_test.values)
pred_dir = np.sign(y_pred)
dir_acc_including_zeros = (actual_dir == pred_dir).mean() * 100
mask_nonzero = y_test.values != 0
dir_acc_excluding_zeros = (actual_dir[mask_nonzero] == pred_dir[mask_nonzero]).mean() * 100 if mask_nonzero.sum() > 0 else np.nan

print(f"  Directional accuracy (including zeros): {dir_acc_including_zeros:.2f}%")
print(f"  Directional accuracy (excluding zero-returns): {dir_acc_excluding_zeros:.2f}%")

# -------------------------
# Compare last 5 days 
# -------------------------
comparison = pd.DataFrame({
    'Actual Return (%)': y_test.tail(5).values,
    'Predicted Return (%)': y_pred[-5:]
}, index=y_test.tail(5).index)
print("\nLast 5 Days Comparison (test set):")
print(comparison)

# -------------------------
# Predict next day (aligned)
# -------------------------
last_feat_idx = X.index[-1]                  # explicit index of last feature row after dropna
last_raw = X.loc[[last_feat_idx]].values     # 2D
last_scaled = scaler.transform(last_raw)
predicted_return = svr.predict(last_scaled)[0]

reference_close = float(data['Close'].loc[last_feat_idx])
predicted_price = reference_close * (1 + predicted_return / 100)

print(f"\nReference date for prediction: {last_feat_idx.date()}")
print(f"Reference close: ${reference_close:.2f}")
print(f"Predicted next day's return (SVR): {predicted_return:.4f} %")
print(f"Predicted next day's closing price (SVR): ${predicted_price:.2f}")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test.values, label='Actual Return', alpha=0.7)
plt.plot(y_test.index, y_pred, label='SVR Predicted Return', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.6)
plt.xlabel('Date'); plt.ylabel('Daily Return (%)')
plt.title(f'{ticker} - Actual vs Predicted Daily Returns (SVR)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,6))
errors = y_test.values - y_pred
plt.scatter(y_test.index, errors, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.xlabel('Date'); plt.ylabel('Prediction Error (%)')
plt.title(f'{ticker} - SVR Prediction Errors (Actual - Predicted)')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
