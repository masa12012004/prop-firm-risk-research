"""Index dip-buying: 6 pre-listed variants, ranked on the DESIGN block only.
lookback k in {1,2,5} days  x  filter in {none, price above 200-day mean}."""
import os, numpy as np, pandas as pd
from pf_lib import *
D = load_daily(); cov = ewcov(D[ASSETS]); r = D[IDX]; vol = ewvol(r)
logp = np.log1p(r.fillna(0)).cumsum()
up = (logp > logp.rolling(200, min_periods=150).mean()).astype(float)
rows, rets = [], {}
for k in (1, 2, 5):
    zk = (r.rolling(k).sum() / (vol * np.sqrt(k))).clip(-2, 2) / 2
    for f in ("none", "ma200"):
        v = pd.DataFrame(0.0, index=D.index, columns=ASSETS)
        s = (-zk).clip(lower=0) * (up if f == "ma200" else 1.0)
        v[IDX] = (s / vol) / len(IDX)
        w = unit_vol(v, cov); net, _, _ = pnl(w, D); g, _, _ = pnl(w, D, 0.0, False)
        name = f"RL{k}_{f}"; rets[name] = net
        rows.append({"variant": name, "DESIGN": sharpe(net["2005":DESIGN_END]),
                     "VALID": sharpe(net[DESIGN_END + pd.Timedelta(days=1):VALID_END]),
                     "BROKER": sharpe(net[BROKER_START:]), "BROKER_gross": sharpe(g[BROKER_START:]),
                     "pct_days_invested": float((w[IDX].abs().sum(axis=1) > 0)["2005":].mean())})
R = pd.DataFrame(rows).round(3); R.to_csv(os.path.join(RES, "p03_reversal_variants.csv"), index=False)
print(R.to_string(index=False))
# sub-period stability of the design-block winner
best = R.sort_values("DESIGN").iloc[-1]["variant"]; x = rets[best]
print("design-block winner:", best)
print(x.groupby(x.index.year).apply(sharpe).round(2).to_string())
