"""FundedHive 2-step challenge run with the gold H1 hybrid (daily F2F regime x hourly cubic), hour by hour.

Input : results/q17_h1_hourly_long.parquet (the Phase-5 code on MT5 XAUUSD H1 2012-2026, monthly walk-forward)
Rules : Phase 1 +8%, Phase 2 +6% (each on a fresh balance); daily loss 5% of the previous end-of-day
        balance; overall loss 10% static; max loss per trade 3%; >= 3 profitable days per phase
        (a day counts at >= PD_THR of the starting balance); no time limit.
Engine: position = LEV x H1 position x balance. One trade = one run of same-sign position.
        Per-trade stop order at TRADE_STOP of starting balance (checked on the bar's high/low, filled at
        the stop plus slippage, or at the open if the bar gaps through it); after a stop-out the account
        stays flat until that signal run ends and until the next trading day.
        Account daily stop at DAY_STOP. Both stops shrink to the room left above the -10% floor.
        The phase ends the moment the target is touched with 3 profitable days banked.
Costs : 0.85 bp half-spread + 0.2 bp slippage per unit turnover, 1.2 bp per night on longs (x3 Wednesday).
Evaluation: a new attempt starts every 5th trading day; each runs until pass, breach, or MAXD trading days.
"""
import os, sys
import numpy as np
import pandas as pd
from pf_lib import Q, RES

TARGET = (0.08, 0.06)
TRADE_STOP, DAY_STOP, FLOOR, BUF = 0.025, 0.04, 0.10, 0.004
PD_THR, PD_NEED, MAXD = 0.01, 3, 504
COST, SWAP, STOP_SLIP = 1.05e-4, 1.2e-4, 5e-4


def load():
    z = pd.read_parquet(os.path.join(RES, "q17_h1_hourly_long.parquet"))
    g = pd.read_csv(os.path.join(Q, "XAUUSDstrat02", "ohlc_xauusd", "XAUUSD_H1.csv"), parse_dates=["time"])
    g["dt"] = (g["time"] - pd.Timedelta(hours=7)).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
    g = g.dropna(subset=["dt"]).drop_duplicates("dt").set_index("dt").sort_index()
    z = z.join(g[["open", "high", "low"]])
    z = z[z["tday"] >= "2019-01-01"]
    return z


def prep(z):
    c = z["close"].values
    pc = np.r_[c[0], c[:-1]]
    day = pd.factorize(z["tday"])[0]
    return dict(pos=z["pos"].values, ro=z["open"].values / pc - 1, rh=z["high"].values / pc - 1,
                rl=z["low"].values / pc - 1, rc=c / pc - 1, day=day,
                newday=np.r_[True, day[1:] != day[:-1]], hr=z.index.hour.values, wd=z.index.weekday.values,
                first=np.r_[0, np.where(day[1:] != day[:-1])[0] + 1], n=len(z))


def attempt(D, i0, lev, trade_stop=TRADE_STOP, day_stop=DAY_STOP, pd_thr=PD_THR):
    """Returns (outcome, trading days used, phase-1 days, min equity, worst day, worst trade).
    outcome: 1 pass both, 2 daily-loss breach, 3 overall breach, 4 trade-loss breach, 0 unfinished."""
    pos, ro, rh, rl, rc, newday, hr, wd, n = (D[k] for k in ("pos", "ro", "rh", "rl", "rc", "newday", "hr", "wd", "n"))
    phase, bal, days, p1d = 0, 1.0, 0, -1
    eod, k_prof = 1.0, 0
    held, tr_pnl, tr_sign, blocked_sign, flat_today = 0.0, 0.0, 0.0, 0.0, False
    min_eq, worst_day, worst_trade = 1.0, 0.0, 0.0
    for t in range(i0 + 1, n):
        if newday[t]:                                         # close the books on the previous day
            dpnl = bal - eod
            worst_day = min(worst_day, dpnl)
            if dpnl >= pd_thr:
                k_prof += 1
            if dpnl <= -0.05 * eod:
                return 2, days, p1d, min_eq, worst_day, worst_trade
            eod, flat_today, days = bal, False, days + 1
            if days >= MAXD:
                return 0, days, p1d, min_eq, worst_day, worst_trade
        # ---- P&L of the position held through bar t (decided at the close of bar t-1)
        if held != 0.0:
            room = bal - (1 - FLOOR) - BUF
            # further loss allowed on this bar: trade stop, daily stop, room above the floor
            s = max(min(trade_stop + tr_pnl, day_stop + (bal - eod), room), 0.0)
            adverse = held * (rl[t] if held > 0 else rh[t])
            need = TARGET[phase] - (bal - 1.0)
            fav = held * (rh[t] if held > 0 else rl[t])
            if adverse <= -s:                                     # stop order
                pnl = min(-s, held * ro[t]) - (STOP_SLIP + COST) * abs(held)   # a gap fills at the open
                bal += pnl; tr_pnl += pnl
                worst_trade = min(worst_trade, tr_pnl)
                if tr_pnl <= -0.03:
                    return 4, days, p1d, min(min_eq, bal), worst_day, worst_trade
                blocked_sign, flat_today, held, tr_pnl = tr_sign, True, 0.0, 0.0
            elif k_prof >= PD_NEED and fav >= need + COST * abs(held):   # take-profit order at the target
                bal += need                                       # (net of the closing cost)
                held, tr_pnl = 0.0, 0.0
            else:
                pnl = held * rc[t]
                if hr[t] == 18 and held > 0:                      # carried through the 17:00 ET roll
                    pnl -= held * SWAP * (3 if wd[t - 1] == 2 else 1)
                bal += pnl; tr_pnl += pnl
        min_eq = min(min_eq, bal)
        if bal <= 1 - FLOOR:
            return 3, days, p1d, min_eq, worst_day, worst_trade
        if bal - 1.0 >= TARGET[phase] - 1e-9 and k_prof + (bal - eod >= pd_thr) >= PD_NEED:
            if phase == 1:
                return 1, days + 1, p1d, min_eq, worst_day, worst_trade
            phase, p1d, bal, eod, k_prof = 1, days + 1, 1.0, 1.0, 0
            held, tr_pnl, tr_sign, blocked_sign, flat_today = 0.0, 0.0, 0.0, 0.0, True
            continue
        # ---- new target position at the close of bar t
        sg = np.sign(pos[t])
        if sg != blocked_sign:
            blocked_sign = 0.0
        want = 0.0 if (flat_today or blocked_sign != 0.0 or bal - (1 - FLOOR) - BUF < 0.003) else lev * pos[t] * bal
        if np.sign(want) != tr_sign:                              # a new trade starts (or the old one ends)
            worst_trade = min(worst_trade, tr_pnl)
            tr_pnl, tr_sign = 0.0, np.sign(want)
        cost = abs(want - held) * COST
        bal -= cost; tr_pnl -= cost if want != 0 else 0.0
        held = want
    return 0, days, p1d, min_eq, worst_day, worst_trade


def run(D, starts, lev, **kw):
    r = np.array([attempt(D, i, lev, **kw) for i in starts])
    o = r[:, 0]
    p = o == 1
    return {"lev": lev, "attempts": len(r), "pass": p.mean(), "breach_daily": (o == 2).mean(),
            "breach_overall": (o == 3).mean(), "breach_trade": (o == 4).mean(), "unfinished": (o == 0).mean(),
            "phase1_passed": (r[:, 2] > 0).mean(),
            "median_days_pass": np.median(r[p, 1]) if p.any() else np.nan,
            "p90_days_pass": np.percentile(r[p, 1], 90) if p.any() else np.nan,
            "pass_by_42": (p & (r[:, 1] <= 42)).mean(), "pass_by_126": (p & (r[:, 1] <= 126)).mean()}


if __name__ == "__main__":
    z = load()
    D = prep(z)
    first = D["first"]
    tdays = z["tday"].values[first]
    pd.set_option("display.width", 250)
    rows = []
    blocks = {"2019-2024 (before the Phase-5 window)": ("2019-01-01", "2023-06-30"),
              "2025-07 onwards (Phase-5 window)": ("2025-07-01", "2026-04-30"),
              "all starts 2019-2026": ("2019-01-01", "2026-04-30")}
    for name, (a, b) in blocks.items():
        m = (tdays >= np.datetime64(a)) & (tdays <= np.datetime64(b))
        starts = first[m][::5]
        for lev in (1, 2, 3, 4, 6, 8):
            rows.append({"start dates": name, **run(D, starts, lev)})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "q18_h1_challenge.csv"), index=False)
    print(R.round(3).to_string(index=False))
