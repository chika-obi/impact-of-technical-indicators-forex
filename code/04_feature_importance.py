"""
Step 4: Extract Random Forest Feature Importance
Identifies which features contribute most to predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Create results directory
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("STEP 4: Feature Importance Analysis (Random Forest)")
print("=" * 60)

# Use EUR/USD as representative pair
ticker = "EURUSD=X"
start_date = "2015-01-01"
end_date = "2024-12-31"

def add_all_indicators(df):
    """Add all 11 technical indicator features"""
    df = df.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    std_dev = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # ATR
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

print(f"\nLoading data for EUR/USD...")

# Download data
df = yf.download(ticker, start=start_date, end=end_date, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

print(f"Downloaded {len(df)} rows")

# Create target
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df[:-1]
print(f"After target creation: {len(df)} rows")

# Add indicators
df = add_all_indicators(df)
df = df.dropna()
print(f"After adding indicators: {len(df)} rows")

# Prepare features
ohlc_features = ['Open', 'High', 'Low', 'Close']
indicator_features = ['MA_5', 'MA_20', 'RSI', 'MACD', 'MACD_signal',
                      'BB_upper', 'BB_lower', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV']

all_features = ohlc_features + indicator_features
X = df[all_features]
y = df['Target']

print(f"Feature set: {len(all_features)} features")
print(f"Features: {all_features}")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

# Extract feature importance
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE RANKINGS")
print("=" * 60)
print(importance_df.to_string(index=False))

# Save to CSV
importance_df.to_csv('results/feature_importance.csv', index=False)
print("\n✅ Feature importance saved to results/feature_importance.csv")

# Summary statistics
ohlc_importance = importance_df[importance_df['Feature'].isin(ohlc_features)]['Importance'].sum()
indicator_importance = importance_df[importance_df['Feature'].isin(indicator_features)]['Importance'].sum()

print("\n" + "=" * 60)
print("IMPORTANCE SUMMARY")
print("=" * 60)
print(f"OHLC features (4): {ohlc_importance*100:.1f}% of total importance")
print(f"Technical indicators (11): {indicator_importance*100:.1f}% of total importance")
print(f"\nMost important feature: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']*100:.1f}%)")
print(f"Least important feature: {importance_df.iloc[-1]['Feature']} ({importance_df.iloc[-1]['Importance']*100:.1f}%)")

# Top 5 and bottom 5
print("\n" + "=" * 60)
print("TOP 5 MOST IMPORTANT FEATURES")
print("=" * 60)
for i in range(5):
    print(f"{i+1}. {importance_df.iloc[i]['Feature']}: {importance_df.iloc[i]['Importance']*100:.2f}%")

print("\n" + "=" * 60)
print("BOTTOM 5 LEAST IMPORTANT FEATURES")
print("=" * 60)
for i in range(1, 6):
    print(f"{i}. {importance_df.iloc[-i]['Feature']}: {importance_df.iloc[-i]['Importance']*100:.2f}%")