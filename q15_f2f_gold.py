"""Forecast-to-Fill (gold folder) implemented from the paper's method section, sections 3-5.
EMA-smoothed log-price slope, standardised on the training window -> p_trend; 50-day momentum flag;
p_bull = 0.6 p_trend + 0.4 m; long when p_bull >= 0.52 and slope > 0; exits: hard stop 2 ATR14,
trailing stop 1.5 ATR14 from the peak, 30-day age limit, p_bear > 0.5; weight = vol-target(15%, cap 2)
x (p_bull - 0.5)/0.5.  Walk-forward: train on the previous (up to) 10 years, trade the next 6 months;
the smoothing parameter is chosen on each training window only.
Data: XAUUSD daily OHLC 2010-08 .. 2026-09 (MT5).  Two execution timings are shown:
  T+1  weight decided at the close of day t earns day t+1      (tradable)
  T+0  weight decided at the close of day t earns day t        (the paper's "same-day" headline; uses the close it trades on)
"""
import os, numpy as np, pandas as pd
from pf_lib import *
g = pd.read_csv(os.path.join(Q, "XAUUSDstrat02", "ohlc_xauusd", "XAUUSD_D1.csv"), parse_dates=["time"]).set_index("time")
g = g[g.index.dayofweek < 5][["open", "high", "low", "close"]]
P = g.close; r = P.pct_change().fillna(0); y = np.log(P)
tr = pd.concat([g.high - g.low, (g.high - P.shift()).abs(), (g.low - P.shift()).abs()], axis=1).max(axis=1)
atr = tr.rolling(14).mean(); mom = (P / P.shift(50) > 1).astype(float)
var = (r ** 2).ewm(alpha=0.06, adjust=False).mean(); wvol = np.minimum(2.0, (0.15 / np.sqrt(252)) / np.sqrt(var))


def run(lam, a, b, mu, sd):
    """Trade rows a..b-1 with slope statistics (mu, sd) frozen from the training window."""
    sl = y.ewm(alpha=1 - lam, adjust=False).mean().diff()
    pb = 0.6 * (((sl - mu) / sd).clip(-3, 3) + 3) / 6 + 0.4 * mom
    w = np.zeros(b - a); inpos = False; ent = peak = 0.0; age = 0
    for i, t in enumerate(range(a, b)):
        c = P.iat[t]
        if inpos:
            age += 1; peak = max(peak, c)
            if c <= ent - 2 * atr.iat[t] or c <= peak - 1.5 * atr.iat[t] or age >= 30 or pb.iat[t] < 0.5: inpos = False
        if not inpos and pb.iat[t] >= 0.52 and sl.iat[t] > 0:
            inpos, ent, peak, age = True, c, c, 0
        if inpos: w[i] = wvol.iat[t] * max(pb.iat[t] - 0.5, 0) / 0.5
    return w


def net(w, lag, cost=COST_SIDE, fin=0.06):
    w = pd.Series(w, index=P.index); wl = w.shift(lag).fillna(0)
    return wl * r - w.diff().abs().shift(lag).fillna(0) * cost - wl.clip(lower=0) * fin / ANN


W = np.zeros(len(P)); chosen = []
start = P.index.searchsorted(P.index[0] + pd.DateOffset(years=3))
for a in range(start, len(P), 126):
    t0 = max(0, P.index.searchsorted(P.index[a] - pd.DateOffset(years=10))); best = (-9, None)
    for lam in (0.90, 0.94, 0.97):
        sl = y.ewm(alpha=1 - lam, adjust=False).mean().diff(); mu, sd = sl.iloc[t0:a].mean(), sl.iloc[t0:a].std()
        wtr = np.zeros(len(P)); wtr[t0 + 60:a] = run(lam, t0 + 60, a, mu, sd)
        s = sharpe(net(wtr, 1).iloc[t0 + 60:a])
        if s > best[0]: best = (s, (lam, mu, sd))
    lam, mu, sd = best[1]; b = min(a + 126, len(P)); W[a:b] = run(lam, a, b, mu, sd); chosen.append(lam)
oos = P.index[start]
print("walk-forward out-of-sample from", oos.date(), "| smoothing chosen:", pd.Series(chosen).value_counts().to_dict(),
      "| days in market %.0f%%" % (100 * (W[start:] > 0).mean()), "| mean weight when in %.2f" % W[W > 0].mean())
rows = []
for lab, lag, c, f in (("T+1, 1 bp/side + 6%/yr financing (my standard)", 1, COST_SIDE, 0.06),
                       ("T+1, paper's cost 0.7 bp, no financing", 1, 0.35e-4, 0.0),
                       ("T+0, paper's cost 0.7 bp, no financing", 0, 0.35e-4, 0.0)):
    x = net(W, lag, c, f)[oos:]
    row = {"timing / costs": lab}
    for pn, (a, b) in {"2013-08..2018": (oos, DESIGN_END), "2019..2021-11": ("2019-01-01", VALID_END), "2021-11..2026-09": (BROKER_START, "2026-12-31"),
                       "paper window 2015..2025": ("2015-01-01", "2025-12-31"), "ALL": (oos, "2026-12-31")}.items():
        row[pn] = sharpe(x[a:b])
    s = stats(x, ci=False); row.update({"ann_ret": s["ann_ret"], "ann_vol": s["ann_vol"], "max_dd": s["max_dd"]}); rows.append(row)
bh = r[oos:]; rows.append({"timing / costs": "gold buy and hold", **{pn: sharpe(bh[a:b]) for pn, (a, b) in
        {"2013-08..2018": (oos, DESIGN_END), "2019..2021-11": ("2019-01-01", VALID_END), "2021-11..2026-09": (BROKER_START, "2026-12-31"),
         "paper window 2015..2025": ("2015-01-01", "2025-12-31"), "ALL": (oos, "2026-12-31")}.items()}, "ann_ret": bh.mean() * ANN, "ann_vol": bh.std() * ANN ** .5,
        "max_dd": stats(bh, ci=False)["max_dd"]})
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "q15_f2f.csv"), index=False)
pd.set_option("display.width", 250); print(R.round(2).to_string(index=False))
x = net(W, 1)[oos:]; lo, hi = sharpe_ci(x.values); print("T+1 standard costs: 90%% interval on ALL-period Sharpe %.2f to %.2f" % (lo, hi))
print("by year:", x.groupby(x.index.year).apply(sharpe).round(2).to_dict())
pd.Series(W, index=P.index, name="w").to_frame().to_parquet(os.path.join(RES, "q15_f2f_weight.parquet"))
