"""Figures for the report (static PNG). Palette: dataviz reference categorical order."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pf_lib import *

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
                     "legend.frameon": False, "savefig.dpi": 140, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


# 1 ---- sleeves, long sample (growth of 1, log scale, each at ~16% annual vol)
S = pd.read_parquet(os.path.join(RES, "p02_sleeve_returns.parquet"))["2005":]
names = {"T": "Trend, all assets", "B": "Long equity indices", "R": "Index 5-day reversal", "RL": "Index dip-buy (5-day)", "GL": "Gold long-only trend"}
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (k, lab) in enumerate(names.items()):
    eq = (1 + S[k]).cumprod(); ax.plot(eq.index, eq, color=C[i], label=f"{lab}  (Sharpe {sharpe(S[k]):.2f})")
for d, t in ((DESIGN_END, "validation starts"), (BROKER_START, "broker data starts")):
    ax.axvline(d, color=INK2, lw=1, ls=":"); ax.text(d, ax.get_ylim()[1], " " + t, fontsize=8, color=INK2, va="top")
ax.set_yscale("log"); ax.set_ylabel("growth of 1 (log)"); ax.legend(loc="upper left", fontsize=9)
ax.set_title("Candidate sleeves after costs and financing, 2005-2026")
save(fig, "fig1_sleeves_long_sample.png")

# 2 ---- the traded book on broker data
from p05_book import build_books, hourly_cube, day_paths
D = load_daily(); H = load_hourly()[ASSETS]; books, _ = build_books(D); days, cube, _ = hourly_cube(H)
wp = books["COMBO"].shift(1).reindex(days).fillna(0.0).to_numpy()
net = pd.Series(day_paths(wp, cube, "swing")[:, -1], index=days)
gross = pd.Series(day_paths(wp, cube, "swing", 0.0, False)[:, -1], index=days)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(days, (1 + gross).cumprod(), color=C[0], label=f"Before costs  (Sharpe {sharpe(gross):.2f})")
ax.plot(days, (1 + net).cumprod(), color=C[1], label=f"After 1 bp/side and 6%/yr financing  (Sharpe {sharpe(net):.2f})")
mlp = os.path.join(RES, "p04_ml_returns.parquet")
if os.path.exists(mlp):
    m = pd.read_parquet(mlp)["ENSEMBLE_ALL_ret"].reindex(days).fillna(0)
    ax.plot(days, (1 + m).cumprod(), color=C[2], label=f"ML/DL forecast ensemble, after costs  (Sharpe {sharpe(m):.2f})")
ax.axvline(pd.Timestamp("2024-07-01"), color=INK2, lw=1, ls=":")
ax.set_ylabel("growth of 1"); ax.legend(loc="upper left", fontsize=9)
ax.set_title("The book (index dip-buy 70% + gold trend 30% of risk) at 1% daily volatility, broker data")
save(fig, "fig2_book_broker_window.png")

# 3 ---- probability of having passed both phases by day N
cdf = json.load(open(os.path.join(RES, "p06_cdf.json")))
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, (s, lab) in enumerate((("zero_edge", "No edge at all"), ("half_edge", "Half the measured edge"), ("measured_edge", "Measured edge"))):
    y = np.array(cdf[f"DP_42d|{s}"]) * 100
    ax.plot(np.arange(1, len(y) + 1), y, color=C[i], label=lab)
    ax.annotate(f"{y[-1]:.0f}%", (len(y), y[-1]), xytext=(5, 0), textcoords="offset points", va="center", color=INK)
ax.set_xlim(1, 45); ax.set_ylim(0, 60); ax.set_xlabel("trading days since the challenge started"); ax.set_ylabel("% of attempts")
ax.legend(loc="upper left"); ax.set_title("Chance that BOTH phases are passed by day N (42-day policy, 20,000 simulated attempts each)")
save(fig, "fig3_pass_probability_by_day.png")

# 4 ---- patience vs probability
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, (p, lab) in enumerate((("DP_42d", "Policy tuned for 42 days"), ("DP_63d", "Tuned for 63 days"), ("DP_126d", "Tuned for 126 days"))):
    y = np.array(cdf[f"{p}|half_edge"]) * 100
    ax.plot(np.arange(1, len(y) + 1), y, color=C[i], label=lab)
    ax.annotate(f"{y[-1]:.0f}%", (len(y), y[-1]), xytext=(5, 0), textcoords="offset points", va="center", color=INK)
ax.axvline(42, color=INK2, lw=1, ls=":"); ax.text(42, 2, " two months", color=INK2, fontsize=8)
ax.set_xlim(1, 134); ax.set_ylim(0, 60); ax.set_xlabel("trading days since the challenge started"); ax.set_ylabel("% of attempts")
ax.legend(loc="upper left"); ax.set_title("More time buys a little more probability (half the measured edge)")
save(fig, "fig4_deadline_tradeoff.png")

# 5 ---- the risk policy (phase 1), sequential single hue
Zp = np.load(os.path.join(RES, "p06_policy_DP_42d.npz")); EG, L = Zp["EG"], Zp["L"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (key, kk, ttl) in zip(axes, (("pol1", 0, "Phase 1, no profitable days banked yet"), ("pol2", 0, "Phase 2, no profitable days banked yet"))):
    A = L[Zp[key][:42, :, kk]].T
    im = ax.imshow(A, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=4, extent=[0, 42, EG[0] * 100, EG[-1] * 100])
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("day of the challenge"); ax.grid(False)
axes[0].set_ylabel("phase profit so far (%)")
cb = fig.colorbar(im, ax=axes, shrink=0.9); cb.set_label("risk level L = targeted daily volatility (%)")
fig.suptitle("The optimal risk policy: how much daily volatility to run in each state", x=0.07, ha="left", fontweight="bold")
save(fig, "fig5_risk_policy.png")

print("figures written:", sorted(os.listdir(FIG)))
