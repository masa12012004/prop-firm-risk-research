"""Build the tradable book and its hour-by-hour P&L on the broker window.

Book      : RL1 (index dip-buy, 1-day) and GL (gold long-only trend), risk-weighted.
Modes     : swing = hold through the roll, pay financing, trade only the weight change
            day   = enter 00:00 UTC, exit 21:00 UTC, flat in between (skips the roll, the reopen
                    gap and every weekend; pays 2 x spread on the full position, no financing;
                    every position is a one-day trade with a hard stop)
Sizing    : weights are rounded to broker lot steps for a $5,000 account, per risk level L
            (L = targeted daily volatility in %).
Output    : results/p05_book.npz  -> used by the challenge simulator.
"""
import os, sys
import numpy as np
import pandas as pd
from pf_lib import *

EQUITY = 5000.0
FLAT_SLOTS = 3
LGRID = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
# lot step and minimum, in USD notional at current prices (Exness contract specs as proxy)
STEP = {"EURUSD": 1116, "GBPUSD": 1340, "AUDUSD": 650, "NZDUSD": 590, "USDCAD": 1000, "USDCHF": 1000, "USDJPY": 1000,
        "SP500": 77, "NAS100": 285, "DOW30": 534, "DAX40": 297, "FTSE100": 147, "JPN225": 4, "XAUUSD": 3900}
MINN = {**{k: v for k, v in STEP.items()}, "SP500": 1080, "NAS100": 1426, "DOW30": 2670, "DAX40": 1865,
        "FTSE100": 735, "JPN225": 1192}


def round_lots(w: pd.DataFrame, equity: float) -> pd.DataFrame:
    out = w.copy()
    for a in w.columns:
        n = w[a] * equity
        q = np.round(n / STEP[a]) * STEP[a]
        q = q.where(q.abs() >= MINN[a], np.where(n.abs() >= 0.5 * MINN[a], np.sign(n) * MINN[a], 0.0))
        out[a] = q / equity
    return out


def rl1(D, vol):
    z1 = (D[IDX] / vol[IDX]).clip(-2, 2) / 2
    v = pd.DataFrame(0.0, index=D.index, columns=ASSETS)
    v[IDX] = ((-z1).clip(lower=0) / vol[IDX]) / len(IDX)
    return v


def build_books(D):
    cov = ewcov(D[ASSETS]); vol = ewvol(D[ASSETS])
    RL = unit_vol(rl1(D, vol), cov)
    GL = unit_vol(raw_sleeves(D)["GL"], cov)
    books = {"RL1": RL, "GL": GL, "COMBO": unit_vol(0.7 * RL + 0.3 * GL, cov)}
    return books, cov


def hourly_cube(H):
    """(day, slot, asset) hourly returns; slot 0 = the bar that opens 21:00 UTC, slot 3 = 00:00 UTC."""
    dk = day_key(H.index)
    days = pd.DatetimeIndex(sorted(set(dk[dk.dayofweek < 5])))
    cube = np.zeros((len(days), 24, H.shape[1]))
    di = days.get_indexer(dk); slot = (H.index.hour - 21) % 24
    ok = di >= 0                 # bars labelled on a weekend day are dropped (none carry returns we hold)
    cube[di[ok], slot[ok]] = H.to_numpy()[ok]
    nh = np.bincount(di[ok], minlength=len(days))
    return days, cube, nh


def day_paths(w_prev: np.ndarray, cube: np.ndarray, mode: str, cost_mult=1.0, financing=True):
    """w_prev[d] = weights held during day d. Returns cumulative hourly P&L path (d, 24)."""
    hp = np.einsum("da,dha->dh", w_prev, cube)
    gross = np.abs(w_prev).sum(axis=1)
    if mode == "day":
        hp[:, :FLAT_SLOTS] = 0.0                                 # flat 21:00-00:00 UTC (roll, reopen gap, weekend)
        hp[:, FLAT_SLOTS] -= gross * COST_SIDE * cost_mult       # enter at 00:00 UTC
        hp[:, -1] -= gross * COST_SIDE * cost_mult               # exit at 21:00 UTC
    else:
        turn = np.abs(np.diff(w_prev, axis=0, prepend=w_prev[:1] * 0)).sum(axis=1)
        hp[:, 0] -= turn * COST_SIDE * cost_mult
        if financing:
            fl = np.array([FIN_LONG[CLASS[a]] for a in ASSETS]); fs = np.array([FIN_SHORT[CLASS[a]] for a in ASSETS])
            hp[:, 0] -= (np.clip(w_prev, 0, None) * fl + np.clip(-w_prev, 0, None) * fs).sum(axis=1) / ANN
    return hp.cumsum(axis=1)


if __name__ == "__main__":
    D = load_daily(); H = load_hourly()[ASSETS]
    books, cov = build_books(D)
    days, cube, nh = hourly_cube(H)
    print("broker days", len(days), "median bars/day", np.median(nh))
    rows = []
    for name, W in books.items():
        wp = W.shift(1).reindex(days).fillna(0.0)
        for mode in ("swing", "day"):
            for label, cm, fin in (("base", 1.0, True), ("gross", 0.0, False), ("cost_x3", 3.0, True)):
                r = pd.Series(day_paths(wp.to_numpy(), cube, mode, cm, fin)[:, -1], index=days)
                s = stats(r, ci=(label == "base"))
                rows.append({"book": name, "mode": mode, "costs": label, **s,
                             "sh_2021_24H1": sharpe(r[:"2024-06-30"]), "sh_2024H2_26": sharpe(r["2024-07-01":]),
                             "gross_lev": float(wp.abs().sum(axis=1).mean())})
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "p05_book_modes.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["days", "skew"]).round(3).to_string(index=False))

    book, mode = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("COMBO", "day")
    W = books[book]; paths = np.zeros((len(LGRID), len(days), 24)); lev = np.zeros((len(LGRID), len(days))); funded = np.zeros_like(paths)
    for i, L in enumerate(LGRID):
        wr = round_lots(W * L, EQUITY).shift(1).reindex(days).fillna(0.0).to_numpy()
        paths[i] = day_paths(wr, cube, mode); lev[i] = np.abs(wr).sum(axis=1)
        # funded stage: no weekend holding -> on Mondays use the flat-through-the-gap path
        pf_ = paths[i].copy(); mon = days.dayofweek == 0
        pf_[mon] = day_paths(wr, cube, "day")[mon]; funded[i] = pf_
    ideal = day_paths(W.shift(1).reindex(days).fillna(0.0).to_numpy(), cube, mode)[:, -1]
    print(f"\n{book}/{mode}: effect of lot rounding on a $5,000 account")
    for i, L in enumerate(LGRID[1:], 1):
        r = paths[i][:, -1]
        print(f"  L={L:4.2f}  daily vol {r.std()*100:5.2f}%  sharpe {sharpe(r):5.2f}  corr with ideal {np.corrcoef(r, ideal)[0,1]:.3f}"
              f"  mean gross lev {lev[i].mean():.2f}  max {lev[i].max():.2f}  days in market {np.mean(lev[i]>0):.2f}")
    np.savez_compressed(os.path.join(RES, "p05_book.npz"), L=LGRID, paths=paths, lev=lev, funded=funded,
                        days=days.values.astype("datetime64[D]").astype(str), book=book, mode=mode)
