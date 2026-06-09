"""
Step 2: Add 11 technical indicators to raw OHLCV data
Indicators: MA_5, MA_20, RSI, MACD, MACD_signal, BB_upper, BB_lower, ATR, Stoch_K, Stoch_D, OBV
"""

import pandas as pd
import numpy as np
import os

def add_all_indicators(df):
    """Add all 11 technical indicator features to dataframe"""
    df = df.copy()
    
    # 1. Moving Averages (2 features)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    
    # 2. RSI (1 feature)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (2 features)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands (2 features)
    df['BB_middle'] = df['Close'].rolling(20).mean()
    std_dev = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # 5. ATR - Average True Range (1 feature)
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # 6. Stochastic Oscillator (2 features)
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # 7. OBV - On-Balance Volume (1 feature)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Create target variable (direction)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df[:-1]  # Remove last row
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Test on EUR/USD as example
print("=" * 60)
print("STEP 2: Adding Technical Indicators (Test on EUR/USD)")
print("=" * 60)

# Load raw data
df_raw = pd.read_csv('../data/EUR_USD.csv', index_col=0, parse_dates=True)

# Add indicators
df_with_indicators = add_all_indicators(df_raw)

print(f"Original rows: {len(df_raw)}")
print(f"After adding indicators and dropping NaN: {len(df_with_indicators)} rows")
print(f"Total features (including OHLC and Target): {len(df_with_indicators.columns)}")
print(f"Technical indicator features: 11")
print("\n✅ Indicator calculation function ready!")

# Save sample for reference
df_with_indicators.head(10).to_csv('../data/sample_with_indicators.csv')
print("✅ Sample saved to data/sample_with_indicators.csv")
