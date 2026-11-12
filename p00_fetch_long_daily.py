"""Fetch long daily history (Yahoo) for signal warm-up and long-sample validation only.
Execution/challenge simulation uses the broker H1 data, never these series."""
import yfinance as yf, pandas as pd
T = {"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","AUDUSD":"AUDUSD=X","NZDUSD":"NZDUSD=X",
     "USDCAD":"USDCAD=X","USDCHF":"USDCHF=X","USDJPY":"USDJPY=X",
     "SP500":"^GSPC","NAS100":"^NDX","DOW30":"^DJI","DAX40":"^GDAXI","FTSE100":"^FTSE",
     "JPN225":"^N225","EU50":"^STOXX50E","CAC40":"^FCHI","AUS200":"^AXJO","XAUUSD":"GC=F"}
out = {}
for k, t in T.items():
    try:
        d = yf.download(t, start="2000-01-01", end="2026-11-08", progress=False, auto_adjust=False)
        c = d["Close"]; c = c.iloc[:, 0] if hasattr(c, "columns") else c
        out[k] = c.dropna(); print(k, len(c), c.index[0].date(), c.index[-1].date())
    except Exception as e:
        print(k, "FAILED", repr(e)[:120])
pd.DataFrame(out).to_parquet("data/yahoo_daily_close.parquet")
