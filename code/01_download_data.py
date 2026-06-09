"""
Step 1: Download all 27 currency pairs from Yahoo Finance
Output: Raw CSV files saved to /data/ directory
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# All 27 actively traded currency pairs
currency_pairs = {
    # Major pairs (7)
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'USDCAD=X',
    'USD/CHF': 'USDCHF=X',
    'NZD/USD': 'NZDUSD=X',
    # Minor pairs (10)
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
    # Exotic pairs (10)
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

print("=" * 60)
print("STEP 1: Downloading Forex Data for 27 Currency Pairs")
print("=" * 60)

for pair_name, ticker in currency_pairs.items():
    print(f"Downloading {pair_name} ({ticker})...")
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Save to CSV
    filename = pair_name.replace('/', '_')
    df.to_csv(f'../data/{filename}.csv')
    
    print(f"  ✅ Saved {len(df)} rows to data/{filename}.csv")

print("\n✅ All data downloaded successfully!")
