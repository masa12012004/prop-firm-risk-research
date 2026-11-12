"""Equity curves and monthly returns of the trade-by-trade learners over the last 12 months of data."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pf_lib import RES, FIG, sharpe, sharpe_ci
R = pd.read_parquet(os.path.join(RES, "q17_daily_returns.parquet"))
cols = {"Hybrid + learning: GRU": "GRU over the last 16 trades", "Hybrid + learning: Average of three": "Average of three learners",
        "Hybrid + learning: LightGBM": "LightGBM learner", "Hybrid: regime x cubic": "Hybrid without learning", "Buy and hold gold": "Gold buy and hold"}
X = R.loc["2025-09-18":"2026-09-17", list(cols)].rename(columns=cols)
M = ((1 + X).groupby(X.index.to_period("M")).prod() - 1) * 100
eq = (1 + X).cumprod()
S = pd.DataFrame({"total %": (eq.iloc[-1] - 1) * 100, "Sharpe": X.apply(sharpe), "Sharpe 90% low": X.apply(lambda c: sharpe_ci(c.values)[0]),
                  "Sharpe 90% high": X.apply(lambda c: sharpe_ci(c.values)[1]), "max drawdown %": (eq / eq.cummax() - 1).min() * 100,
                  "months up": (M > 0).sum(), "months down": (M < 0).sum(), "best month %": M.max(), "worst month %": M.min(),
                  "share of profit from best 2 months %": M.apply(lambda c: c.nlargest(2).sum() / c.sum() * 100)})
M.round(2).to_csv(os.path.join(RES, "q18_monthly_last12m.csv")); S.round(2).to_csv(os.path.join(RES, "q18_summary_last12m.csv"))
pd.set_option("display.width", 220); print(M.round(1).to_string()); print(S.round(2).T.to_string())
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8983"]; INK2, SURF, GRID = "#52514e", "#fcfcfb", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 1.3]}, sharex=True)
for i, c in enumerate(eq):
    ax[0].plot(eq.index, eq[c], color=C[i], lw=1.2 if i == 4 else 2, label=f"{c}  ({(eq[c].iloc[-1]-1)*100:+.0f}%, Sharpe {sharpe(X[c]):.2f})")
    if i < 4: ax[1].plot(eq.index, (eq[c] / eq[c].cummax() - 1) * 100, color=C[i], lw=1.5)
ax[0].axhline(1, color=INK2, lw=0.8); ax[0].set_ylabel("growth of 1.0"); ax[0].legend(loc="upper left", fontsize=9)
ax[0].set_title("Gold hybrid with trade-by-trade learning, 18 Sep 2025 to 17 Sep 2026, net of spread and slippage")
ax[1].set_ylabel("drawdown, %"); ax[1].set_title("Drawdown (strategies only)", fontsize=10)
fig.savefig(os.path.join(FIG, "fig7_learners_last12m.png"), dpi=140, bbox_inches="tight")
