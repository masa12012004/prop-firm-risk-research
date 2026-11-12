"""Audit of the gold hybrid + learners backtest, long/short split, and the GRU <-> Average-of-three frontier.

Needs q17_positions.parquet (full run) and q17_positions_cut202603.parquet (same code, data truncated at 2026-03-01).
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pf_lib import RES, FIG, sharpe
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
P = pd.read_parquet(os.path.join(RES, "q17_positions.parquet")); PC = pd.read_parquet(os.path.join(RES, "q17_positions_cut202603.parquet"))
STR = {"Hybrid": "Hybrid, no learning", "GRU": "GRU (last 16 trades)", "Average of three": "Average of three", "LightGBM": "LightGBM"}
tday = pd.Series((P.index + pd.Timedelta(hours=7)).normalize().tz_localize(None), index=P.index)
ret = P["ret"]; C0 = 0.85e-4 + 0.2e-4
roll = pd.Series(np.where(P.index.hour == 16, np.where(P.index.dayofweek == 2, 3.0, 1.0), 0.0), index=P.index)   # 17:00 NY rollover, triple on Wednesday

def hourly(pos, cost_mult=1.0, delay=0, fin_long=0.0, fin_short=0.0):
    pos = pos.shift(delay).fillna(0.0) if delay else pos
    return pos * ret.shift(-1).fillna(0.0) - pos.diff().abs().fillna(pos.abs()) * C0 * cost_mult \
           - roll * (pos.clip(lower=0) * fin_long + (-pos).clip(lower=0) * fin_short) / 365.0
def daily(x): return x.groupby(tday).sum()
WIN = {"last 12 months": ("2025-09-18", "2026-09-17"), "last 3 years": ("2023-10-01", "2026-09-17"), "Jul 2020 - Sep 2023": ("2020-07-01", "2023-09-30")}

# ------------------------------------------------------------ 1. reproducibility and look-ahead
R = pd.read_parquet(os.path.join(RES, "q17_daily_returns.parquet")); R0 = pd.read_parquet(os.path.join(RES, "q17_daily_returns_prev.parquet"))
print("== 1a. rerun reproduces the earlier run: max |difference| in daily returns:", float((R - R0).abs().max().max()))
for k in STR:
    col = "Hybrid: regime x cubic" if k == "Hybrid" else f"Hybrid + learning: {k}"
    d = (daily(hourly(P[k])).reindex(R.index).fillna(0) - R[col])["2019-07-02":].abs().max()
    print(f"   independent P&L recomputation from saved positions, {k}: max |diff| {d:.2e}")
print("== 1b. truncation test: positions from data ending 2026-02-28 vs the full run, on the same bars")
rows = []
for k in ["regime", "cubic", "Hybrid", "Online logistic", "LightGBM", "GRU", "Average of three"]:
    a, b = PC[k], P[k].reindex(PC.index); nz = (a != 0) | (b != 0)
    rows.append({"series": k, "bars compared": len(a), "bars in a position": int(nz.sum()), "max |diff|": float((a - b).abs().max()),
                 "bars where the sign differs": int((np.sign(a) != np.sign(b)).sum()), "median ratio cut/full": float((a[nz] / b[nz]).replace([np.inf, -np.inf], np.nan).median())})
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------ 2. stress tests
rows = []
tests = {"as reported": {}, "enter one hour late": {"delay": 1}, "enter two hours late": {"delay": 2}, "costs x2": {"cost_mult": 2}, "costs x3": {"cost_mult": 3},
         "swap: longs pay 6%/yr": {"fin_long": 0.06}, "swap: longs 6%, shorts 2%/yr": {"fin_long": 0.06, "fin_short": 0.02},
         "one hour late + costs x2 + swap": {"delay": 1, "cost_mult": 2, "fin_long": 0.06, "fin_short": 0.02}}
D = {}
for t, kw in tests.items():
    for k in STR:
        d = daily(hourly(P[k], **kw)); D[(t, k)] = d
        for w, (a, b) in WIN.items():
            x = d[a:b]; rows.append({"test": t, "strategy": STR[k], "window": w, "total %": ((1 + x).prod() - 1) * 100, "Sharpe": sharpe(x)})
ST = pd.DataFrame(rows); ST.round(2).to_csv(os.path.join(RES, "q19_stress.csv"), index=False)
print("\n== 2. stress tests, Sharpe"); print(ST.pivot_table(index=["test"], columns=["window", "strategy"], values="Sharpe", sort=False).round(2).to_string())
print("\n   total return %"); print(ST.pivot_table(index=["test"], columns=["window", "strategy"], values="total %", sort=False).round(1).to_string())

# ------------------------------------------------------------ 3. long vs short
rows = []
for k in STR:
    pos = P[k]; h = hourly(pos); sg = np.sign(P["Hybrid"]); ent = (sg != sg.shift(1)) & (sg != 0)
    for w, (a, b) in WIN.items():
        m = (tday >= a) & (tday <= b)
        rows.append({"strategy": STR[k], "window": w, "hours long %": (pos[m] > 0).mean() * 100, "hours short %": (pos[m] < 0).mean() * 100, "hours flat %": (pos[m] == 0).mean() * 100,
                     "long trades": int((ent & (sg > 0) & m).sum()), "short trades": int((ent & (sg < 0) & m).sum()),
                     "P&L from longs %": h[m & (pos > 0)].sum() * 100, "P&L from shorts %": h[m & (pos < 0)].sum() * 100})
LS = pd.DataFrame(rows); LS.round(2).to_csv(os.path.join(RES, "q19_long_short.csv"), index=False); print("\n== 3. long vs short (P&L is the simple sum of hourly net returns)"); print(LS.round(1).to_string(index=False))

# ------------------------------------------------------------ 4. frontier GRU <-> Average of three, and a switching rule
G, A = D[("as reported", "GRU")], D[("as reported", "Average of three")]
def st(x):
    eq = (1 + x).cumprod(); return {"ann return %": x.mean() * 252 * 100, "ann vol %": x.std() * np.sqrt(252) * 100, "Sharpe": sharpe(x), "max drawdown %": (eq / eq.cummax() - 1).min() * 100}
rows = []
for w, (a, b) in WIN.items():
    for wt in np.round(np.arange(0, 1.01, 0.1), 2):
        rows.append({"window": w, "weight on GRU": wt, **st((wt * G + (1 - wt) * A)[a:b])})
FR = pd.DataFrame(rows); FR.round(3).to_csv(os.path.join(RES, "q19_frontier.csv"), index=False)
print("\n== 4. frontier"); print(FR.round(2).to_string(index=False))
print("correlation of daily returns GRU vs Average:", {w: round(float(G[a:b].corr(A[a:b])), 3) for w, (a, b) in WIN.items()})
# switching: each month hold whichever had the higher Sharpe over the previous 3 / 6 months (known at the prior month end)
rows = []
for look in (3, 6):
    per = G.index.to_period("M"); pick = pd.Series(index=G.index, dtype=float)
    for m in per.unique():
        end = m.to_timestamp(); beg = (m - look).to_timestamp(); g_, a_ = G[beg:end - pd.Timedelta(days=1)], A[beg:end - pd.Timedelta(days=1)]
        pick[per == m] = 1.0 if (len(g_) > 20 and sharpe(g_) > sharpe(a_)) else 0.0
    sw = pick * G + (1 - pick) * A
    for w, (a, b) in WIN.items(): rows.append({"rule": f"switch on trailing {look}-month Sharpe", "window": w, "share of months in GRU %": pick[a:b].mean() * 100, **st(sw[a:b])})
for nm, x in (("always GRU", G), ("always Average of three", A), ("fixed 50/50", 0.5 * G + 0.5 * A)):
    for w, (a, b) in WIN.items(): rows.append({"rule": nm, "window": w, "share of months in GRU %": np.nan, **st(x[a:b])})
SW = pd.DataFrame(rows); SW.round(3).to_csv(os.path.join(RES, "q19_switching.csv"), index=False); print(SW.round(2).sort_values(["window", "rule"]).to_string(index=False))

# ------------------------------------------------------------ figure
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK, INK2, SURF, GRID = "#1f1f1d", "#52514e", "#fcfcfb", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for i, (w, (a, b)) in enumerate(WIN.items()):
    f = FR[FR.window == w]; ax[i].plot(f["ann vol %"], f["ann return %"], color=INK2, lw=1.5, marker="o", ms=3, zorder=2)
    pts = {"GRU (last 16 trades)": (st(G[a:b]), C[0]), "Average of three": (st(A[a:b]), C[1]), "LightGBM": (st(D[("as reported", "LightGBM")][a:b]), C[2]),
           "Hybrid, no learning": (st(D[("as reported", "Hybrid")][a:b]), C[3])}
    for nm, (s, c) in pts.items():
        ax[i].scatter(s["ann vol %"], s["ann return %"], color=c, s=70, zorder=3)
        ax[i].annotate(f"{nm}\nSharpe {s['Sharpe']:.2f}", (s["ann vol %"], s["ann return %"]), textcoords="offset points", xytext=(8, {"GRU (last 16 trades)": -26, "Average of three": 8}.get(nm, -4)), fontsize=8.5, color=INK)
    ax[i].set_title(w); ax[i].set_xlabel("volatility, % a year"); ax[i].margins(0.25)
    if i == 0: ax[i].set_ylabel("return, % a year")
fig.suptitle("Blends of the GRU and Average-of-three learners (grey line, 10% steps) against the other versions", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_frontier_gru_vs_average.png"), dpi=140, bbox_inches="tight")
