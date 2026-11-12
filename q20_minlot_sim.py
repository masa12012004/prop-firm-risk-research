"""What the live cBot actually trades on a $5k account: fixed 0.01 lot (1 oz) per entry, direction from
the GRU-sized hybrid, entry only when |w| >= threshold, bracket SL = 1% of equity / TP = 2% of equity,
exit otherwise when the signal flips or goes flat. Hourly path on the audited positions; SL/TP checked
on hourly high/low. Compared with the continuous-size backtest."""
import os, numpy as np, pandas as pd
from pf_lib import RES, sharpe
P = pd.read_parquet(os.path.join(RES, "q17_positions.parquet"))
g = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "XAUUSDstrat02", "ohlc_xauusd", "XAUUSD_H1.csv"), parse_dates=["time"])
g["dt"] = (g["time"] - pd.Timedelta(hours=7)).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
g = g.dropna(subset=["dt"]).drop_duplicates("dt").set_index("dt").sort_index().reindex(P.index)
hi, lo, cl = g["high"].values, g["low"].values, g["close"].values
EQ0, OZ, COST = 5000.0, 1.0, 1.05e-4
tday = (P.index + pd.Timedelta(hours=7)).normalize().tz_localize(None)

def sim(w, thr, sl_pct=0.01, tp_pct=0.02, oz=OZ, max_per_day=3):
    sg = np.sign(w.values); mag = np.abs(w.values); n = len(sg)
    eq = EQ0; side = 0; entry = 0.0; eqs = np.full(n, np.nan); day_prev = None; nt = 0; trades = []
    for t in range(n - 1):
        d = tday[t]
        if d != day_prev: day_prev, nt = d, 0
        want = sg[t] if mag[t] >= thr else 0
        if side != 0 and want != side:                                # signal exit / flip at close t
            pnl = side * (cl[t] - entry) * oz - 2 * COST * entry * oz; eq += pnl; trades.append(pnl); side = 0
        if side == 0 and want != 0 and nt < max_per_day:
            side, entry, nt = want, cl[t], nt + 1
            sl = entry - side * sl_pct * eq / oz; tp = entry + side * tp_pct * eq / oz
        if side != 0:                                                 # bracket check on next bar's range
            h, l = hi[t + 1], lo[t + 1]
            hit_sl = (l <= sl) if side > 0 else (h >= sl); hit_tp = (h >= tp) if side > 0 else (l <= tp)
            if hit_sl or hit_tp:
                px = sl if hit_sl else tp                              # conservative: SL first if both
                pnl = side * (px - entry) * oz - 2 * COST * entry * oz; eq += pnl; trades.append(pnl); side = 0
        eqs[t] = eq
    e = pd.Series(eqs, index=P.index).ffill().groupby(tday).last()
    r = e.pct_change().fillna(0.0); tr = np.array(trades)
    return r, tr

def report(r, tr, name):
    out = {}
    for lab, a, b in (("last 12m", "2025-09-18", "2026-09-17"), ("last 3y", "2023-10-01", "2026-09-17"), ("2020-07..2023-09", "2020-07-01", "2023-09-30")):
        x = r[a:b]; eq = (1 + x).cumprod()
        out[lab] = f"{(eq.iloc[-1]-1)*100:+6.1f}%  Sh {sharpe(x):5.2f}  maxDD {(eq/eq.cummax()-1).min()*100:6.1f}%"
    print(f"{name:34s}", " | ".join(f"{k}: {v}" for k, v in out.items()), f"| trades {len(tr)}, win {np.mean(tr>0)*100:.0f}%")

print("Fixed 0.01 lot on $5,000, SL 1% / TP 2% of equity, max 3 entries/day  (returns compounded on equity)")
for thr in (0.02, 0.05, 0.1, 0.2, 0.4):
    r, tr = sim(P["GRU"], thr); report(r, tr, f"GRU gate |w|>={thr}")
r, tr = sim(P["Hybrid"], 0.05); report(r, tr, "Hybrid (no learner) gate 0.05")
r, tr = sim(P["GRU"], 0.05, sl_pct=0.01, tp_pct=0.03); report(r, tr, "GRU gate 0.05, TP 3%")
r, tr = sim(P["GRU"], 0.05, sl_pct=0.01, tp_pct=9.0); report(r, tr, "GRU gate 0.05, no TP")
print("\nContinuous-size backtest (what the audit measured), daily returns:")
ret = P["ret"]
for k in ("GRU", "Hybrid"):
    pos = P[k]; net = (pos * ret.shift(-1).fillna(0) - pos.diff().abs().fillna(0) * COST).groupby(tday).sum()
    report(net, np.array([0.0]), f"continuous {k}")
