"""Challenge and funded-stage statistics for the book chosen by the no-look rule (q13: H1),
plus the position-sizing rules proposed in the sizing papers, all under the same simulator.
  Kelly / fractional Kelly (3, 5, 6, 16, 41)      drawdown-modulated sizing (52, 13, 26, 4)
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
from pf_lib import *
from p05_book import hourly_cube, day_paths, round_lots, LGRID, EQUITY
NPZ = os.path.join(RES, "q14_book.npz")
if not os.path.exists(NPZ) or "--rebuild" in sys.argv:
    W = pd.read_pickle(os.path.join(RES, "q13_weights.pkl"))["H1 dip-buy + gold January (70/30 risk)"]
    days, cube, _ = hourly_cube(load_hourly()[ASSETS])
    paths = np.zeros((len(LGRID), len(days), 24)); lev = np.zeros((len(LGRID), len(days))); funded = np.zeros_like(paths)
    for i, L in enumerate(LGRID):
        wr = round_lots(W * L, EQUITY).shift(1).reindex(days).fillna(0.0).to_numpy()
        paths[i] = day_paths(wr, cube, "swing"); lev[i] = np.abs(wr).sum(axis=1)
        f_ = paths[i].copy(); mon = days.dayofweek == 0; f_[mon] = day_paths(wr, cube, "day")[mon]; funded[i] = f_
    np.savez_compressed(NPZ, L=LGRID, paths=paths, lev=lev, funded=funded, days=days.values.astype("datetime64[D]").astype(str), book="H1", mode="swing")
os.environ["PF_BOOK"] = NPZ
import p06_challenge as C
ND, SPLIT, PATHS, LG = C.ND, C.SPLIT, C.PATHS, C.LG
train, test = np.arange(0, SPLIT), np.arange(SPLIT, ND)
r1 = PATHS[4][:, -1]
print("book H1 at L=1 after lot rounding: Sharpe all %.2f | train %.2f | test %.2f | mean bp/day %.2f" %
      (sharpe(r1), sharpe(r1[train]), sharpe(r1[test]), r1.mean() * 1e4))
scen = {"zero_edge": 0.0, "half_edge": 0.5, "measured_edge": 1.0}; adj = {s: C.drift_adjust(PATHS, k) for s, k in scen.items()}
rows, cdf = [], {}

def run(name, c, pf, T):
    for seg, idx in (("hist_test_2024H2_2026", test), ("hist_train_2021_2024H1", train)):
        st = idx[: len(idx) - T]; seq = st[:, None] + np.arange(T)[None, :]
        rows.append({"policy": name, "eval": seg, **C.summarise(C.simulate(c, PATHS, seq, pf), T)})
    seq = C.boot_seq(20000, T, np.arange(ND), 10, 11)
    for s in scen:
        res = C.simulate(c, adj[s], seq, pf); rows.append({"policy": name, "eval": "boot_" + s, **C.summarise(res, T)})
        cdf[f"{name}|{s}"] = [float(((res["done"] == 1) & (res["day"] <= m)).mean()) for m in range(1, T + 1)]

for T in (42, 126):
    c = C.cfg(T=T); dp = C.solve_dp(c, adj["zero_edge"], train); run(f"DP_{T}d", c, C.policy_fn(c, dp), T)
c = C.cfg(T=42)
mu, sd = PATHS[4][train, -1].mean(), PATHS[4][train, -1].std(); kelly = max(mu / sd ** 2 / 100, 0.0)       # in L units
print("Kelly-optimal daily vol from TRAIN: L = %.2f (mean %.2f bp, sd %.2f%%)" % (kelly, mu * 1e4, sd * 100))
nearest = lambda L: int(np.argmin(np.abs(LG - L)))
run("Full Kelly (constant L=%.2f)" % LG[nearest(kelly)], c, lambda ph, t, e, k: np.full(len(e), nearest(kelly)), 42)
run("Half Kelly (constant L=%.2f)" % LG[nearest(kelly / 2)], c, lambda ph, t, e, k: np.full(len(e), nearest(kelly / 2)), 42)
def ddmod(kappa):
    grid = LG[None, :]
    return lambda ph, t, e, k: np.abs(grid - (kappa * (e + 0.10))[:, None]).argmin(1)
run("Drawdown-modulated, L = 30 x cushion", c, ddmod(30), 42)
run("Drawdown-modulated, L = 40 x cushion", c, ddmod(40), 42)
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "q14_pass_probabilities.csv"), index=False)
json.dump(cdf, open(os.path.join(RES, "q14_cdf.json"), "w"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["policy", "eval", "p_pass", "p_fail_daily", "p_fail_total", "p_timeout", "p_phase1", "median_p1_days", "median_days_if_pass", "mean_L", "pass_by_21", "pass_by_30", "pass_by_42"]
print(R[cols].round(3).to_string(index=False))
env = dict(os.environ, PF_FUNDED_OUT="q14_funded.csv")
print(subprocess.run([sys.executable, "-W", "ignore", "p07_funded.py"], env=env, capture_output=True, text=True).stdout)
