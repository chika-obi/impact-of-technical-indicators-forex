"""
Step 3: Run ablation experiments - Baseline (OHLC only) vs Full Set (OHLC + 11 indicators)
Models: Logistic Regression, Random Forest, SVM
Validation: 5-fold Time Series Cross-Validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Create results directory
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("STEP 3: Running Ablation Experiments")
print("27 Currency Pairs | 5-fold Time Series CV")
print("=" * 60)

# All 27 currency pairs
currency_pairs = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'USDCAD=X',
    'USD/CHF': 'USDCHF=X',
    'NZD/USD': 'NZDUSD=X',
    'EUR/GBP': 'EURGBP=X',
    'EUR/JPY': 'EURJPY=X',
    'EUR/CHF': 'EURCHF=X',
    'EUR/AUD': 'EURAUD=X',
    'EUR/CAD': 'EURCAD=X',
    'GBP/JPY': 'GBPJPY=X',
    'GBP/CHF': 'GBPCHF=X',
    'GBP/AUD': 'GBPAUD=X',
    'GBP/CAD': 'GBPCAD=X',
    'AUD/JPY': 'AUDJPY=X',
    'USD/ZAR': 'USDZAR=X',
    'USD/SEK': 'USDSEK=X',
    'USD/NOK': 'USDNOK=X',
    'USD/MXN': 'USDMXN=X',
    'USD/TRY': 'USDTRY=X',
    'USD/SGD': 'USDSGD=X',
    'USD/HKD': 'USDHKD=X',
    'USD/HUF': 'USDHUF=X',
    'USD/PLN': 'USDPLN=X',
    'USD/CZK': 'USDCZK=X'
}

start_date = "2015-01-01"
end_date = "2024-12-31"

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

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

def evaluate_model_cv(X, y, model, n_splits=5):
    """Time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)
        
        scores.append(accuracy_score(y_test, y_pred))
    
    return np.mean(scores), np.std(scores)

# Main experiment
results = []

for pair_name, ticker in currency_pairs.items():
    print(f"\nProcessing {pair_name}...")
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Create target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df[:-1]
    
    # Add indicators
    df = add_all_indicators(df)
    df = df.dropna()
    
    ohlc_features = ['Open', 'High', 'Low', 'Close']
    indicator_features = ['MA_5', 'MA_20', 'RSI', 'MACD', 'MACD_signal',
                          'BB_upper', 'BB_lower', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV']
    
    X_baseline = df[ohlc_features]
    X_full = df[ohlc_features + indicator_features]
    y = df['Target']
    
    # Evaluate Random Forest (primary model)
    rf_baseline, _ = evaluate_model_cv(X_baseline, y, models['Random Forest'])
    rf_full, _ = evaluate_model_cv(X_full, y, models['Random Forest'])
    
    results.append({
        'Currency': pair_name,
        'Baseline_Accuracy': f"{rf_baseline*100:.2f}%",
        'Full_Set_Accuracy': f"{rf_full*100:.2f}%",
        'Difference': f"{(rf_full - rf_baseline)*100:+.2f}pp",
        'Winner': 'Baseline' if rf_full < rf_baseline else 'Full Set'
    })
    
    print(f"  Baseline: {rf_baseline*100:.2f}% | Full Set: {rf_full*100:.2f}% | Diff: {(rf_full - rf_baseline)*100:+.2f}pp")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/baseline_vs_full_results.csv', index=False)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

baseline_wins = (results_df['Winner'] == 'Baseline').sum()
print(f"\n✅ Baseline wins: {baseline_wins} out of {len(results_df)} pairs")
print(f"✅ Results saved to results/baseline_vs_full_results.csv")