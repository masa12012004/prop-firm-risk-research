"""Hybrids built ONLY from what cleared the pre-declared bar on DESIGN and VALID
(net Sharpe >= 0.3 on DESIGN and > 0 on VALID).  Hybrids are ranked on DESIGN+VALID; the BROKER
block is reported afterwards and plays no part in the choice."""
import os, numpy as np, pandas as pd
from pf_lib import *
from p05_book import rl1
D = load_daily()[ASSETS]; cov = ewcov(D); vol = ewvol(D)
T = pd.read_csv(os.path.join(RES, "q10_daily_battery.csv"))
ok = T[(T.DESIGN >= 0.3) & (T.VALID > 0)]
print("sleeves clearing the bar:\n", ok[["strategy", "universe", "DESIGN", "VALID"]].round(2).to_string(index=False))
W = pd.read_pickle(os.path.join(RES, "q10_weights.pkl"))
dip = unit_vol(rl1(D, vol), cov)
jan = W["S12b gold January long only|GOLD"]; gtr = W["REF gold long-only Baz trend|GOLD"]
# S16: HAR volatility forecast (1, 5, 22 day) fitted on an expanding window, replacing EWMA in the sizing
r2 = (D[IDX] ** 2)
X = pd.concat({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()}, axis=1)
har = pd.DataFrame(index=D.index, columns=IDX, dtype=float)
for a in IDX:
    x = np.column_stack([np.ones(len(D)), X["d"][a], X["w"][a], X["m"][a]]); yv = r2[a].shift(-1).values
    for t0 in range(1000, len(D), 250):
        m = np.isfinite(x[:t0 - 1]).all(1) & np.isfinite(yv[:t0 - 1])
        b = np.linalg.lstsq(x[:t0 - 1][m], yv[:t0 - 1][m], rcond=None)[0]
        har.iloc[t0:t0 + 250, IDX.index(a)] = np.sqrt(np.clip(x[t0:t0 + 250] @ b, 1e-8, None))
volh = vol.copy(); volh[IDX] = har.fillna(vol[IDX])
# causal HMM exposure from q10 (re-derive from the saved S7 / S11 weights)
s11, s7 = W["S11 long-only stock+gold, vol-controlled|ALL"], W["S7 same with HMM volatility-regime exposure|ALL"]
expo = (s7["SP500"] / s11["SP500"].replace(0, np.nan)).clip(0, 3).fillna(1.0); expo = expo / expo.expanding().mean()
rhat = (D - D.expanding(250).mean()) / vol.shift(1); q = np.exp(-2 / 128); phi = rhat.fillna(0).ewm(alpha=1 - q, adjust=False).mean() * np.sqrt(1 - np.exp(-4 / 128)) / (1 - q)
gate = (phi[IDX] > -1.44).astype(float).reindex(columns=ASSETS).fillna(1.0)

H = {"H0 index dip-buy alone": dip,
     "H1 dip-buy + gold January (70/30 risk)": unit_vol(0.7 * dip + 0.3 * jan, cov),
     "H2 dip-buy x HMM volatility-regime exposure": dip.mul(expo.clip(0.25, 1.5), axis=0),
     "H3 dip-buy sized with HAR volatility forecast": unit_vol(rl1(D, volh), cov),
     "H4 dip-buy, skipped when cubic trend is in the left tail": unit_vol(dip * gate, cov),
     "H5 dip-buy + gold January + HAR sizing": unit_vol(0.7 * unit_vol(rl1(D, volh), cov) + 0.3 * jan, cov),
     "REF first-study book: dip-buy + gold trend (70/30)": unit_vol(0.7 * dip + 0.3 * gtr, cov)}
rows, rets = [], {}
for k, w in H.items():
    net, _, _ = pnl(w, D); g, _, _ = pnl(w, D, 0.0, False); rets[k] = net
    rows.append({"hybrid": k, "DESIGN": sharpe(net["2005":DESIGN_END]), "VALID": sharpe(net[DESIGN_END + pd.Timedelta(days=1):VALID_END]),
                 "DESIGN+VALID": sharpe(net["2005":VALID_END]), "BROKER": sharpe(net[BROKER_START:]), "BROKER_gross": sharpe(g[BROKER_START:]),
                 "BROKER_ann_ret": net[BROKER_START:].mean() * ANN, "BROKER_maxdd": stats(net[BROKER_START:], ci=False)["max_dd"]})
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "q13_hybrids.csv"), index=False)
pd.to_pickle(H, os.path.join(RES, "q13_weights.pkl")); pd.DataFrame(rets).to_parquet(os.path.join(RES, "q13_returns.parquet"))
pd.set_option("display.width", 220); print(R.round(3).to_string(index=False))
best = R[~R.hybrid.str.startswith("REF")].sort_values("DESIGN+VALID").iloc[-1]
print("\nchosen on DESIGN+VALID:", best.hybrid)
lo, hi = sharpe_ci(rets[best.hybrid][BROKER_START:].values); print("its BROKER Sharpe %.2f, 90%% interval %.2f to %.2f" % (best.BROKER, lo, hi))
x = rets[best.hybrid][BROKER_START:]; print("by year:", x.groupby(x.index.year).apply(sharpe).round(2).to_dict())
