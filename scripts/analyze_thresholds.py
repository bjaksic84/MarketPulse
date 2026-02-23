#!/usr/bin/env python3
"""Quick analysis of return distributions to find balanced thresholds."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime, timedelta
from src.data.fetcher import YFinanceFetcher
from src.data.market_config import load_market_config
from src.data.preprocessing import preprocess_ohlcv

cases = [
    ('indices', '^GSPC', 'S&P500'),
    ('stocks', 'AAPL', 'Apple'),
    ('stocks', 'MSFT', 'Microsoft'),
    ('crypto', 'BTC-USD', 'Bitcoin'),
    ('crypto', 'SOL-USD', 'Solana'),
    ('futures', 'GC=F', 'Gold'),
    ('futures', 'CL=F', 'Oil'),
]

end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

print(f"{'Ticker':<12} {'1d_std':>8} {'3d_std':>8} | {'1d_p33':>8} {'3d_p33':>8}")
print('-' * 55)

for market, ticker, label in cases:
    mc = load_market_config(market)
    f = YFinanceFetcher(market_config=mc)
    raw = f.fetch(ticker, start=start, end=end)
    df = preprocess_ohlcv(raw, market_config=mc)
    
    r1 = df['close'].pct_change().dropna()
    r3 = (df['close'] / df['close'].shift(3) - 1).dropna()
    
    t1 = np.percentile(r1.abs(), 33)
    t3 = np.percentile(r3.abs(), 33)
    
    print(f'{label:<12} {r1.std():>8.4f} {r3.std():>8.4f} | {t1:>8.4f} {t3:>8.4f}')
    
    for h, ret, lbl in [(1, r1, '1d'), (3, r3, '3d')]:
        t = np.percentile(ret.abs(), 33)
        up = (ret > t).mean()
        flat = ((ret >= -t) & (ret <= t)).mean()
        down = (ret < -t).mean()
        print(f'  {lbl} @{t:.3%}: DOWN={down:.1%} FLAT={flat:.1%} UP={up:.1%}')
