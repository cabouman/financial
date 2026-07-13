"""Fetch monthly total-return data and build spliced log-wealth series.

Assets (in fixed order): CASH, FLCOX, FSKAX, FXNAX, FTIHX
  CASH  = 3-month T-bill wealth index (FRED TB3MS) standing in for FDRXX,
          whose $1 NAV series carries no distribution information.
Backfill proxies (used only before each fund's inception, spliced in
log-return space; any window touching proxy data is labeled 'spliced'):
  FSKAX <- VTSMX (Vanguard Total Stock Mkt Idx, 1992)
  FLCOX <- VIVAX (Vanguard Value Index, 1992)
  FTIHX <- VGTSX (Vanguard Total Intl Stock Idx, 1996)
  FXNAX <- VBMFX (Vanguard Total Bond Mkt Idx, 1986) — Yahoo's FXNAX history starts 2011.
Output: data/monthly_log_returns.csv (joint monthly log returns),
        data/splice_dates.csv (first real-fund month per asset).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FUNDS = ["FLCOX", "FSKAX", "FXNAX", "FTIHX"]
PROXY = {"FSKAX": "VTSMX", "FLCOX": "VIVAX", "FTIHX": "VGTSX", "FXNAX": "VBMFX"}

tickers = FUNDS + [p for p in PROXY.values() if p]
data = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)
px = data["Close"].resample("ME").last()
logret = np.log(px).diff()

# Cash: TB3MS annualized percent -> monthly log return
urllib.request.urlretrieve("https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS", "data/tb3ms.csv")
tb = pd.read_csv("data/tb3ms.csv", parse_dates=[0], index_col=0)
tb.index = tb.index + pd.offsets.MonthEnd(0)
cash = np.log((1 + tb.iloc[:, 0] / 100.0) ** (1 / 12))
cash.name = "CASH"

# Splice each fund with its proxy before inception
out = {"CASH": cash}
splice = {}
for f in FUNDS:
    fund = logret[f].dropna()
    splice[f] = fund.index[0]
    if PROXY[f]:
        proxy = logret[PROXY[f]].dropna()
        pre = proxy[proxy.index < fund.index[0]]
        out[f] = pd.concat([pre, fund])
    else:
        out[f] = fund

df = pd.DataFrame(out)[["CASH", "FLCOX", "FSKAX", "FXNAX", "FTIHX"]]
df = df.loc[:"2026-06-30"]  # drop partial current month
df.to_csv("data/monthly_log_returns.csv")
pd.Series(splice).to_csv("data/splice_dates.csv")

print("Series coverage (monthly log returns):")
for c in df.columns:
    s = df[c].dropna()
    tag = f" (real fund from {splice[c]:%Y-%m})" if c in splice else ""
    print(f"  {c}: {s.index[0]:%Y-%m} to {s.index[-1]:%Y-%m}, {len(s)} months{tag}")
common = df.dropna()
print(f"Common window: {common.index[0]:%Y-%m} to {common.index[-1]:%Y-%m} ({len(common)} months = {len(common)/12:.1f} years)")
