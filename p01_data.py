"""Build the two return panels used everywhere.

hourly_ret.parquet : broker H1 simple returns on a common UTC grid, 2021-11 -> 2026-10.
                     FX majors + index CFDs (forexStrategyDesign) + XAUUSD (xauusd_system
                     UTC H1 to 2026-08, HistData M1->H1 afterwards).
daily_ret.parquet  : daily returns. Yahoo closes before the broker window (warm-up and
                     long-sample validation), broker H1 aggregated to 21:00-UTC days after.
"""
import os
import numpy as np
import pandas as pd
from pf_lib import Q, DATA, FX, IDX, GOLD, ASSETS, BROKER_START, day_key

END = pd.Timestamp("2026-10-30 21:00")


def mt5_h1(path):
    d = pd.read_csv(path, usecols=["time", "close"], parse_dates=["time"])
    return d.drop_duplicates("time").set_index("time")["close"].sort_index()


closes = {}
for p in FX:
    closes[p] = mt5_h1(os.path.join(Q, "forexStrategyDesign", "data", "H1", f"{p}_H1.csv"))
for p in IDX:
    closes[p] = mt5_h1(os.path.join(Q, "forexStrategyDesign", "data_indices", "H1", f"{p}_H1.csv"))

# gold: UTC broker H1, then HistData (tz-aware ET) for the tail
g1 = pd.read_csv(os.path.join(Q, "xauusd_system", "results_phase0", "raw", "XAUUSD_H1.csv"),
                 usecols=["time", "close"])
# The file labels bars "+00:00" but the hour-of-day volatility profile shows MT5 server
# time (New York + 7h: dead hour at 00:00, US open at 15:30-16:30). Convert to true UTC.
g1["time"] = (pd.to_datetime(g1["time"], utc=True).dt.tz_localize(None) - pd.Timedelta(hours=7))
g1["time"] = (g1["time"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
              .dt.tz_convert("UTC").dt.tz_localize(None))
g1 = g1.dropna(subset=["time"]).drop_duplicates("time")
g1 = g1.set_index("time")["close"]
g2 = pd.read_parquet(os.path.join(Q, "gold_hist_Data", "clean_Data", "xauusd_h1.parquet"),
                     columns=["dt_et", "close"])
g2["time"] = pd.to_datetime(g2["dt_et"], utc=True).dt.tz_localize(None)
g2 = g2.set_index("time")["close"]
ov = g1.index.intersection(g2.index)
print(f"gold overlap bars {len(ov)}, hourly-return corr "
      f"{np.corrcoef(g1[ov].pct_change().dropna(), g2[ov].pct_change().dropna())[0, 1]:.4f}, "
      f"median level ratio {(g1[ov] / g2[ov]).median():.5f}")
cut = g1.index[-1]
closes["XAUUSD"] = pd.concat([g1, g2[g2.index > cut] * (g1[ov] / g2[ov]).median()])

grid = pd.date_range("2021-11-04 21:00", END, freq="h")
C = pd.DataFrame({k: v.reindex(grid.union(v.index)).ffill().reindex(grid) for k, v in closes.items()})[ASSETS]
# keep only hours in which at least one market printed a bar
live = pd.DataFrame({k: v.reindex(grid).notna() for k, v in closes.items()}).any(axis=1)
C = C[live]
H = C.pct_change().iloc[1:]
H = H[H.index >= pd.Timestamp("2021-11-07 21:00")]
print("hourly", H.shape, H.index[0], H.index[-1])
print("stale share per asset (zero hourly return):")
print((H == 0).mean().round(3).to_string())
big = H.abs().max()
print("max |hourly ret|:", big.round(4).to_dict())
H.to_parquet(os.path.join(DATA, "hourly_ret.parquet"))

# daily from hourly
dk = day_key(H.index)
Db = (1 + H).groupby(dk).prod() - 1
Db = Db[Db.index.dayofweek < 5]
print("broker days", len(Db), Db.index[0].date(), Db.index[-1].date())

Y = pd.read_parquet(os.path.join(DATA, "yahoo_daily_close.parquet"))[ASSETS]
# Yahoo stamps an FX bar with the date on which it OPENS; shift to the closing date
# (verified: correlation with broker days is 0.95 at this shift, 0.04 without it).
Y[FX] = Y[FX].shift(-1)
Yr = Y.ffill(limit=3).pct_change(fill_method=None)
Yr = Yr[(Yr.index >= "2004-01-01") & (Yr.index < Db.index[0])]
Yr = Yr[Yr.index.dayofweek < 5]
# sanity: Yahoo vs broker daily correlation inside the broker window
Yb = Y.ffill(limit=3).pct_change(fill_method=None).reindex(Db.index)
print("Yahoo vs broker daily-return corr:")
print(pd.Series({a: Yb[a].corr(Db[a]) for a in ASSETS}).round(3).to_string())
D = pd.concat([Yr, Db]).clip(-0.25, 0.25)
D.to_parquet(os.path.join(DATA, "daily_ret.parquet"))
print("daily", D.shape, D.index[0].date(), D.index[-1].date())
print("ann vol %:", (D.std() * np.sqrt(252) * 100).round(1).to_dict())
