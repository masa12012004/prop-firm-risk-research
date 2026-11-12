"""Hourly strategies from the papers on broker H1 data (14 instruments, 2021-11 .. 2026-10).
TRAIN 2021-11 .. 2024-06 (anything fitted is fitted here) | TEST 2024-07 .. 2026-10.
Decide at the close of hour t, earn hour t+1, 1 bp per side. No financing charged (short holds)."""
import os, warnings
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
from pf_lib import *
warnings.filterwarnings("ignore")
H = load_hourly()[ASSETS]; SPLIT = pd.Timestamp("2024-07-01")
hv = np.sqrt((H ** 2).ewm(span=240, min_periods=100).mean()).shift(1)       # hourly vol, lagged
z1 = (H / hv).clip(-10, 10)
S = {}

# S13 push-response / drift-burst reversal (92, 102): fade large hourly pushes, hold 4 hours
for k in (1, 4):
    zk = H.rolling(k).sum() / (hv * np.sqrt(k))
    for name, sig in ((f"S13 buy after {k}h push < -2 sd", (zk < -2).astype(float)),
                      (f"S13 fade both sides of {k}h push > 2 sd", -np.sign(zk) * (zk.abs() > 2))):
        S[name] = sig.rolling(4).max().where(sig.rolling(4).min() >= 0, sig.rolling(4).sum().clip(-1, 1))
    S[f"S13 follow {k}h push > 2 sd (drift-burst continuation)"] = (np.sign(zk) * (zk.abs() > 2)).rolling(4).sum().clip(-1, 1)

# S14 intraday HMM momentum (75): 2-state Gaussian HMM per asset, fitted on TRAIN, forward filter only
hm = pd.DataFrame(0.0, index=H.index, columns=ASSETS)
for a in ASSETS:
    x = z1[a].fillna(0).values.reshape(-1, 1); ntr = int((H.index < SPLIT).sum())
    m = GaussianHMM(2, covariance_type="diag", n_iter=30, random_state=0).fit(x[:ntr])
    A, mu, sd = m.transmat_, m.means_.ravel(), np.sqrt(m.covars_.ravel()); al = m.startprob_.copy(); out = np.zeros(len(x))
    for t in range(len(x)):
        lik = np.exp(-0.5 * ((x[t, 0] - mu) / sd) ** 2) / sd + 1e-300
        al = (al @ A) * lik; al /= al.sum(); out[t] = (al @ A) @ mu
    hm[a] = np.sign(out) * (np.abs(out) > np.abs(out[:ntr]).mean())
S["S14 HMM intraday momentum"] = hm

# S15 directional change (24): threshold = half a daily sd; follow the confirmed trend, or fade it
dc = pd.DataFrame(0.0, index=H.index, columns=ASSETS)
lp = np.log1p(H.fillna(0)).cumsum()
for a in ASSETS:
    p = lp[a].values; th = (hv[a] * np.sqrt(24) * 0.5).bfill().values; mode = 1; ext = p[0]; out = np.zeros(len(p))
    for t in range(len(p)):
        if mode == 1:
            ext = max(ext, p[t])
            if p[t] <= ext - th[t]: mode, ext = -1, p[t]
        else:
            ext = min(ext, p[t])
            if p[t] >= ext + th[t]: mode, ext = 1, p[t]
        out[t] = mode
    dc[a] = out
S["S15 directional change, follow trend"] = dc
S["S15 directional change, contrarian"] = -dc

rows = []
dk = day_key(H.index)
for name, sig in S.items():
    for uni, cols in (("ALL", ASSETS), ("FX", FX), ("IDX", IDX), ("GOLD", GOLD)):
        w = (sig[cols].fillna(0) / (hv[cols] * np.sqrt(24))) / len(cols) * 0.01
        g = (w.shift(1) * H[cols]).sum(axis=1); c = w.diff().abs().shift(1).sum(axis=1) * COST_SIDE
        gd, nd = g.groupby(dk).sum(), (g - c).groupby(dk).sum()
        gd, nd = gd[gd.index.dayofweek < 5], nd[nd.index.dayofweek < 5]
        rows.append({"strategy": name, "universe": uni, "TRAIN": sharpe(nd[:SPLIT]), "TRAIN_gross": sharpe(gd[:SPLIT]),
                     "TEST": sharpe(nd[SPLIT:]), "TEST_gross": sharpe(gd[SPLIT:]),
                     "trades_per_day": float(w.diff().abs().gt(0).sum(axis=1).groupby(dk).sum().mean())})
        if uni == "ALL": nd.to_frame(name).to_parquet(os.path.join(RES, f"q11_{abs(hash(name)) % 10**8}.parquet"))
T = pd.DataFrame(rows); T.to_csv(os.path.join(RES, "q11_intraday_battery.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); print(T.round(2).to_string(index=False))
