"""Long-history check of the Phase-5 gold winners, using the Phase-5 code itself.

The functions are imported unchanged from gold_hist_Data/phase5_literature (s01_f2f.build,
s02_cubic_tsmom.cubic_positions, common.backtest/vol_target).  Only the data and the out-of-sample
window change: MT5 XAUUSD H1 bars 2010-08 .. 2026-09 instead of 2025-01 .. 2026-10, monthly refits
from 2012-01, constant half-spread 0.85 bp (the Phase-5 real-spread figure) + 0.2 bp slippage.
Nothing is written into the Phase-5 results folder.
"""
import os, sys, numpy as np, pandas as pd
P5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gold_hist_Data", "phase5_literature")
sys.path.insert(0, P5)
import common, s01_f2f, s02_cubic_tsmom
from pf_lib import Q, RES, sharpe, sharpe_ci

START, END = pd.Timestamp("2012-01-01", tz="US/Eastern"), pd.Timestamp("2026-09-18", tz="US/Eastern")
common.OOS_START, common.OOS_END = START, END
def refits(index, start=START, end=END):
    idx = index[(index >= start) & (index < end)]
    return [pd.Timestamp(x) for x in pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first().tolist()]
s01_f2f.monthly_refit_points = refits; s02_cubic_tsmom.monthly_refit_points = refits

g = pd.read_csv(os.path.join(Q, "XAUUSDstrat02", "ohlc_xauusd", "XAUUSD_H1.csv"), parse_dates=["time"])
g["dt"] = (g["time"] - pd.Timedelta(hours=7)).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
g = g.dropna(subset=["dt"]).drop_duplicates("dt").set_index("dt").sort_index()
h1 = g[["open", "high", "low", "close"]].copy(); h1["mid"] = h1["close"]
h1["ret"] = h1["mid"].pct_change().fillna(0.0); h1["lret"] = np.log(h1["mid"]).diff().fillna(0.0)
h1["hs"] = 0.85e-4; h1["tday"] = (h1.index + pd.Timedelta(hours=7)).normalize().tz_localize(None)
gb = h1.groupby("tday")
dly = pd.DataFrame({"open": gb["open"].first(), "high": gb["high"].max(), "low": gb["low"].min(), "close": gb["close"].last(), "n": gb.size()})
dly = dly[dly.n > 12]; dly["hs"] = 0.85e-4
dly.index = pd.DatetimeIndex(dly.index, name="dt").tz_localize("US/Eastern") + pd.Timedelta(hours=16, minutes=55)
dly["mid"] = dly["close"]; dly["ret"] = dly["mid"].pct_change().fillna(0.0); dly["lret"] = np.log(dly["mid"]).diff().fillna(0.0)
dly["tday"] = dly.index.tz_localize(None).normalize()
print("H1 bars", len(h1), h1.index[0], h1.index[-1], "| daily bars", len(dly))

bpy = common.bars_per_year(h1)
fc, _ = s02_cubic_tsmom.cubic_positions(h1)
cubic = common.vol_target(fc.clip(-1, 1), h1["ret"], bpy, 0.15, span=24 * 20)
f2f_ls, _ = s01_f2f.build(dly, True); f2f_lo, _ = s01_f2f.build(dly, False)
key = pd.DataFrame({"t": (h1.index + pd.Timedelta(hours=1)).tz_convert("UTC")}, index=h1.index).reset_index()
dk = pd.DataFrame({"t": (dly.index + pd.Timedelta(minutes=5)).tz_convert("UTC"), "f2f": f2f_ls}).sort_values("t")
regime = pd.merge_asof(key.sort_values("t"), dk, on="t").set_index("dt")["f2f"].reindex(h1.index).fillna(0.0)
hyb = cubic.where((np.sign(cubic) == np.sign(regime)) & (regime != 0), 0.0)

def daily(bt, shift=False):
    r = bt.groupby("tday")["net"].sum(); return r.shift(1).fillna(0.0) if shift else r
R = pd.DataFrame({"Buy and hold gold": daily(common.backtest(dly, np.ones(len(dly)), "x", save=False), True),
                  "F2F regime, long-only (daily)": daily(common.backtest(dly, f2f_lo, "x", save=False), True),
                  "F2F regime, long/short (daily)": daily(common.backtest(dly, f2f_ls, "x", save=False), True),
                  "Cubic trend-reversion (H1)": daily(common.backtest(h1, cubic, "x", save=False)),
                  "H1 hybrid: regime x cubic": daily(common.backtest(h1, hyb, "x", save=False))}).fillna(0.0)
R.to_parquet(os.path.join(RES, "q16_gold_long_history.parquet"))
blocks = {"2012-2018": ("2012", "2018"), "2019-2021": ("2019", "2021"), "2022 - 2025-06": ("2022", "2025-06-30"),
          "2025-07 - 2026-09 (Phase-5 window)": ("2025-07-01", "2026-09-30"), "ALL 2012-2026": ("2012", "2026-09-30")}
T = pd.DataFrame({b: {c: sharpe(R[c][a:z]) for c in R} for b, (a, z) in blocks.items()})
T["ann_ret"] = R.mean() * 252; T["max_dd"] = ((1 + R).cumprod() / (1 + R).cumprod().cummax() - 1).min()
T.to_csv(os.path.join(RES, "q16_gold_long_history.csv")); pd.set_option("display.width", 250); print(T.round(2).to_string())
h = R["H1 hybrid: regime x cubic"]; lo, hi = sharpe_ci(h.values)
print("hybrid ALL-period 90%% interval: %.2f to %.2f" % (lo, hi)); print("hybrid by year:", h.groupby(h.index.year).apply(sharpe).round(2).to_dict())
print("hybrid share of bars in market: %.0f%%" % (100 * (hyb[hyb.index >= START] != 0).mean()))
