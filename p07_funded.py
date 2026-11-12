"""Funded stage: what the account is worth after passing.

FundedHive assigns a risk category from how the challenge was traded; the category sets the
funded-account limits and its price (rules PDF, pages 2-3):
  Low      1% / trade, 2% daily, 5% max drawdown, fee 1% of capital taken from profits
  Moderate 1.5% / 3% / 6%,                         fee 2% from profits
  Medium   2% / 4% / 7%,                           fee 2.5% paid up front
  High     3% / 5% / 10%,                          fee 3% paid up front
No weekend holding. 80% profit split. Payout cadence is not stated in the PDF: a withdrawal of all
profit above the starting balance every 21 trading days is assumed.
"""
import os, numpy as np, pandas as pd
from pf_lib import RES, block_boot_idx
Z = np.load(os.environ.get("PF_BOOK", os.path.join(RES, "p05_book.npz")), allow_pickle=True)
LG, P, LEV = Z["L"], Z["funded"], Z["lev"]; NL, ND, NH = P.shape
CAP = 5000.0
CATS = {"Low": dict(dstop=0.010, dd=0.05, daily=0.02, fee=0.01, upfront=False),
        "Moderate": dict(dstop=0.015, dd=0.06, daily=0.03, fee=0.02, upfront=False),
        "Medium": dict(dstop=0.020, dd=0.07, daily=0.04, fee=0.025, upfront=True),
        "High": dict(dstop=0.030, dd=0.10, daily=0.05, fee=0.03, upfront=True)}


def run(cat, Li, keep, n=10000, T=252, seed=5):
    c = CATS[cat]; rng = np.random.default_rng(seed)
    mu = P[Li, :, -1].mean(); ramp = (np.arange(NH) + 1) / NH
    Pa = P[Li] - (1 - keep) * mu * ramp[None, :]
    seq = np.stack([block_boot_idx(ND, T, 10, rng) for _ in range(n)])
    e = np.zeros(n); alive = np.ones(n, bool); paid = np.zeros(n); npay = np.zeros(n)
    for t in range(T):
        d = seq[:, t]; path = Pa[d]
        S = np.minimum(c["dstop"], e + c["dd"] - 0.003)
        tr = alive & (S > 0.002)
        hit = path <= -np.where(tr, S, 1e9)[:, None]
        hs = np.where(hit.any(1), hit.argmax(1), NH)
        pnl = path[:, -1].copy(); st = hs < NH
        hc = path[np.arange(n), np.minimum(hs, NH - 1)]
        rt = np.maximum(hc, -S - LEV[Li, d] * 5e-4)
        pnl[st] = np.where(hs[st] < 3, hc[st], rt[st])
        pnl = np.where(tr, pnl, 0.0)
        dead = alive & ((pnl <= -c["daily"]) | (e + pnl <= -c["dd"]))
        e = e + pnl; alive &= ~dead
        if (t + 1) % 21 == 0:
            w = alive & (e > 0)
            paid[w] += 0.8 * e[w] * CAP; npay[w] += 1; e[w] = 0.0
    fee = c["fee"] * CAP
    net = paid - (fee if c["upfront"] else np.minimum(fee, paid))
    return dict(category=cat, L=LG[Li], edge=keep, ann_vol_pct=P[Li, :, -1].std() * np.sqrt(252) * 100,
                p_breach_or_stuck_12m=float(((~alive) | (e + c['dd'] < 0.01)).mean()), p_any_payout=(npay > 0).mean(), mean_net_usd=net.mean(),
                median_net_usd=np.median(net), p10_net_usd=np.percentile(net, 10), p90_net_usd=np.percentile(net, 90))


rows = []
for cat, Ls in (("Low", (0.25, 0.5, 0.75)), ("Moderate", (0.5, 0.75, 1.0)), ("High", (0.75, 1.0, 1.5, 2.0))):
    for L in Ls:
        for keep in (0.0, 0.5, 1.0):
            rows.append(run(cat, int(np.argmin(np.abs(LG - L))), keep))
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, os.environ.get("PF_FUNDED_OUT", "p07_funded.csv")), index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
