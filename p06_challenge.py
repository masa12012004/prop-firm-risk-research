"""FundedHive 2-step challenge: exact-rule simulator, optimal risk policy, pass probabilities.

Rules simulated (from the rules PDF + the user's rule table)
  Phase 1 target +8%, Phase 2 target +6%, measured on the phase's starting balance
  Daily loss 5% of the previous end-of-day balance  (checked on FLOATING equity -> conservative)
  Overall loss 10% of initial, static
  Max loss per trade 3%  -> the whole book carries an account-level hard stop <= 3% per day and every
                            position is a one-day trade, so no single trade can lose more than the stop
  >= 3 profitable days per phase (a day counts if it earns >= PD_THR of initial; 1% assumed = strict)
  No time limit from the firm; the user's limit is 42 trading days (two months) for both phases.

Policy   : each day choose the risk level L (targeted daily vol, %) from the state
           (phase, day number, equity, profitable days). Solved by backward dynamic programming on
           TRAIN days (2021-11 .. 2024-06) with the drift REMOVED, so the policy does not lean on
           an edge that may not be there.
Testing  : (a) every historical start date in the untouched second half (2024-07 .. 2026-10)
           (b) stationary block bootstrap of the full window under three edge scenarios.
"""
import os, sys, json
import numpy as np
import pandas as pd
from pf_lib import RES, block_boot_idx

Z = np.load(os.environ.get("PF_BOOK", os.path.join(RES, "p05_book.npz")), allow_pickle=True)
LG, PATHS, DAYS, LEV = Z["L"], Z["paths"], pd.DatetimeIndex(Z["days"]), Z["lev"]
GAP_SLOTS = 3
SLIP_BP = 5e-4      # stop-out slippage per unit of gross leverage (real-time equity stop)
NL, ND, NH = PATHS.shape
SPLIT = int(np.searchsorted(DAYS, pd.Timestamp("2024-07-01")))
TARGET = {1: 0.08, 2: 0.06}
EG_STEP = 0.001


def cfg(**kw):
    c = dict(stop="realtime", dstop=0.03, floor=-0.10, daily_lim=0.05, buf=0.004, pd_thr=0.01, pd_need=3, T=42, lmax=4.0)
    c.update(kw); return c


def drift_adjust(paths, keep):
    """Scale the average daily drift of every risk level to `keep` x measured (0 = no edge)."""
    mu = paths[:, :, -1].mean(axis=1)
    ramp = (np.arange(NH) + 1) / NH
    return paths - (1 - keep) * mu[:, None, None] * ramp[None, None, :]


def day_outcome(P, S, need, tp_on, slip=None):
    """Hour-by-hour resolution of one day. P (n,24) cumulative P&L; S stop (n,), need (n,).
    Stop is checked first within an hour. Returns realised P&L."""
    idx = np.arange(len(P))
    hit_s = P <= -S[:, None]
    hs = np.where(hit_s.any(1), hit_s.argmax(1), NH)
    hit_t = (P >= need[:, None]) & tp_on[:, None]
    ht = np.where(hit_t.any(1), hit_t.argmax(1), NH)
    out = P[:, -1].copy()
    st = hs <= np.minimum(ht, NH - 1)
    # realtime: a cBot flattens the book when floating loss reaches the stop (fill = stop + slippage,
    #           never better than the stop and never worse than the hour's close)
    # hourly  : stop is only acted on at the close of the piercing hour (very conservative)
    hc = P[idx[st], hs[st]]
    rt = -S[st] - (0.0 if slip is None else slip[st])
    # slots 0-2 (21:00-00:00 UTC) hold the roll / reopen / weekend gap: a stop cannot beat a gap
    out[st] = hc if slip is None else np.where(hs[st] < GAP_SLOTS, hc, np.maximum(hc, rt))
    tp = (~st) & (ht < NH)
    out[tp] = need[tp]                                  # limit order at the target
    return out


def step(c, phase, e, k, a, P, slip=None):
    """Advance one day for n accounts. Returns new (phase, e, k) and fail flags."""
    tgt = np.where(phase == 1, TARGET[1], TARGET[2])
    cushion = e - c["floor"]
    S = np.minimum(c["dstop"], cushion - c["buf"])
    trade = (S > 0.002) & (a > 0)
    S = np.where(trade, S, 1e9)
    need = tgt - e
    tp_on = (k >= c["pd_need"]) | ((k == c["pd_need"] - 1) & (need >= c["pd_thr"]))
    pnl = np.where(trade, day_outcome(P, S, np.maximum(need, 1e-6), tp_on, slip if c['stop'] == 'realtime' else None), 0.0)
    fail_daily = pnl <= -c["daily_lim"] * (1 + e)
    e2 = e + pnl
    fail_total = e2 <= -0.10
    k2 = np.minimum(k + (pnl >= c["pd_thr"]), c["pd_need"])
    passed = (e2 >= tgt - 1e-9) & (k2 >= c["pd_need"]) & ~fail_daily & ~fail_total
    return e2, k2, passed, fail_daily, fail_total, pnl


# ----------------------------------------------------------------------------- DP
def solve_dp(c, paths, train_idx):
    """V[phase][t, e, k] = P(pass both phases by day T). Returns policy[phase][t, e, k] -> L index."""
    T = c["T"]; EG = np.round(np.arange(c["floor"], 0.08 + 1e-9, EG_STEP), 4); ne = len(EG)
    acts = [i for i, L in enumerate(LG) if L <= c["lmax"]]
    nk = c["pd_need"] + 1
    Ptr = paths[:, train_idx, :]; n = len(train_idx); Str = LEV[:, train_idx] * SLIP_BP
    # outcome tensor O[phase][a, e, k] -> (n,) new equity, new k, pass, fail   (time independent)
    trans = {}
    for ph in (1, 2):
        tr = {}
        for a in acts:
            for kk in range(nk):
                e = np.repeat(EG, n); k = np.full(ne * n, kk)
                P = np.tile(Ptr[a], (ne, 1))
                e2, k2, ps, fd, ft, _ = step(c, np.full(ne * n, ph), e, k, np.full(ne * n, a), P, np.tile(Str[a], ne))
                fail = fd | ft
                ei = np.clip(np.round((e2 - c["floor"]) / EG_STEP).astype(int), 0, ne - 1)
                tr[(a, kk)] = (ei.reshape(ne, n), k2.reshape(ne, n), ps.reshape(ne, n), fail.reshape(ne, n))
        trans[ph] = tr
    V = {ph: np.zeros((T + 2, ne, nk)) for ph in (1, 2)}
    pol = {ph: np.zeros((T + 1, ne, nk), dtype=np.int8) for ph in (1, 2)}
    i0 = int(round((0 - c["floor"]) / EG_STEP))
    for t in range(T - 1, -1, -1):
        for ph in (2, 1):
            best = np.full((ne, nk), -1.0); arg = np.zeros((ne, nk), dtype=np.int8)
            for a in acts:
                for kk in range(nk):
                    ei, k2, ps, fail = trans[ph][(a, kk)]
                    cont = V[ph][t + 1][ei, k2]
                    win = 1.0 if ph == 2 else V[2][t + 1][i0, 0]
                    val = np.where(fail, 0.0, np.where(ps, win, cont)).mean(axis=1)
                    # prefer the smaller risk level on ties (acts ascend)
                    upd = val > best[:, kk] + 1e-9
                    best[upd, kk] = val[upd]; arg[upd, kk] = a
            V[ph][t] = best; pol[ph][t] = arg
    return {"V": V, "pol": pol, "EG": EG, "v0": float(V[1][0][i0, 0])}


def policy_fn(c, dp=None, const=None):
    if const is not None:
        ai = int(np.argmin(np.abs(LG - const)))
        return lambda phase, t, e, k: np.full(len(e), ai)
    EG, pol = dp["EG"], dp["pol"]

    def f(phase, t, e, k):
        ei = np.clip(np.round((e - c["floor"]) / EG_STEP).astype(int), 0, len(EG) - 1)
        tt = min(t, c["T"])
        return np.where(phase == 1, pol[1][tt][ei, k], pol[2][tt][ei, k])
    return f


# ----------------------------------------------------------------------------- simulation
def simulate(c, paths, seq, pf, horizon=None, record=False):
    """seq (n, Tmax) day indices. Returns dict of outcome arrays."""
    n, Tm = seq.shape; Tm = horizon or Tm
    phase = np.ones(n, int); e = np.zeros(n); k = np.zeros(n, int)
    done = np.zeros(n, int)        # 0 running, 1 passed both, 2 daily breach, 3 total breach
    day_done = np.full(n, -1); p1_day = np.full(n, -1); Luse = np.zeros(n); Lcnt = np.zeros(n)
    maxdd = np.zeros(n); worst = np.zeros(n)
    eq = np.zeros((n, Tm + 1)) if record else None; ph_rec = np.ones((n, Tm + 1), int) if record else None
    for t in range(Tm):
        live = done == 0
        if not live.any(): break
        ii = np.where(live)[0]
        a = pf(phase[ii], t, e[ii], k[ii])
        P = paths[a, seq[ii, t], :]
        e2, k2, ps, fd, ft, pnl = step(c, phase[ii], e[ii], k[ii], a, P, LEV[a, seq[ii, t]] * SLIP_BP)
        Luse[ii] += LG[a]; Lcnt[ii] += 1
        worst[ii] = np.minimum(worst[ii], pnl); maxdd[ii] = np.minimum(maxdd[ii], e2)
        e[ii], k[ii] = e2, k2
        done[ii[fd]] = 2; done[ii[ft & ~fd]] = 3
        day_done[ii[fd | ft]] = t + 1
        p = ii[ps]
        fin = p[phase[p] == 2]; adv = p[phase[p] == 1]
        done[fin] = 1; day_done[fin] = t + 1
        p1_day[adv] = t + 1; phase[adv] = 2; e[adv] = 0.0; k[adv] = 0
        if record:
            eq[:, t + 1] = np.where(phase == 2, TARGET[1] + e, e); ph_rec[:, t + 1] = phase
            eq[done == 1, t + 1] = TARGET[1] + TARGET[2]
    return dict(done=done, day=day_done, p1_day=p1_day, meanL=Luse / np.maximum(Lcnt, 1), phase=phase,
                e=e, worst=worst, maxdd=maxdd, eq=eq)


def boot_seq(n_paths, T, pool, block, seed):
    rng = np.random.default_rng(seed)
    return np.stack([pool[block_boot_idx(len(pool), T, block, rng)] for _ in range(n_paths)])


def summarise(res, T, marks=(10, 15, 21, 30, 42, 63, 84, 126)):
    d, day = res["done"], res["day"]; n = len(d)
    out = {"n": n, "p_pass": float((d == 1).mean()), "p_fail_daily": float((d == 2).mean()),
           "p_fail_total": float((d == 3).mean()), "p_timeout": float((d == 0).mean()),
           "p_phase1": float(((res["p1_day"] > 0)).mean()),
           "median_days_if_pass": float(np.median(day[d == 1])) if (d == 1).any() else np.nan,
           "median_p1_days": float(np.median(res["p1_day"][res["p1_day"] > 0])) if (res["p1_day"] > 0).any() else np.nan,
           "mean_L": float(res["meanL"].mean()), "worst_day_p5": float(np.percentile(res["worst"], 5))}
    for m in marks:
        if m <= T: out[f"pass_by_{m}"] = float(((d == 1) & (day <= m)).mean())
    return out


if __name__ == "__main__":
    NPATH = 20000
    train = np.arange(0, SPLIT); test = np.arange(SPLIT, ND)
    scen = {"zero_edge": 0.0, "half_edge": 0.5, "measured_edge": 1.0}
    adj = {s: drift_adjust(PATHS, kp) for s, kp in scen.items()}
    mu = PATHS[:, :, -1]
    print(f"book={Z['book']} mode={Z['mode']} days={ND} train={len(train)} test={len(test)}")
    print("unit-risk (L=1) daily mean bp: train %.2f  test %.2f | daily sd %%: %.2f" %
          (mu[4, train].mean() * 1e4, mu[4, test].mean() * 1e4, mu[4].std() * 100))
    rows, cdf, keep = [], {}, {}
    variants = {
        "DP_42d": cfg(T=42),
        "DP_63d": cfg(T=63),
        "DP_126d": cfg(T=126),
        "DP_42d_lowrisk_style": cfg(T=42, dstop=0.02, floor=-0.05),   # behaves like the "Low Risk" funded category
        "DP_126d_lowrisk_style": cfg(T=126, dstop=0.02, floor=-0.05),
        "DP_42d_hourly_stop": cfg(T=42, stop="hourly"),               # sensitivity: stop acted on once an hour
    }
    for name, c in variants.items():
        dp = solve_dp(c, adj["zero_edge"], train); pf = policy_fn(c, dp); keep[name] = (c, dp)
        print(f"{name}: DP value (train, zero edge) = {dp['v0']:.3f}", flush=True)
        T = c["T"]
        # (a) historical rolling starts, untouched test half and train half
        for seg, idx in (("hist_test_2024H2_2026", test), ("hist_train_2021_2024H1", train)):
            st = idx[: len(idx) - T]
            seq = st[:, None] + np.arange(T)[None, :]
            r = summarise(simulate(c, PATHS, seq, pf), T)
            rows.append({"policy": name, "eval": seg, **r})
        # (b) block bootstrap over the full window
        seq = boot_seq(NPATH, T, np.arange(ND), 10, 11)
        for s in scen:
            res = simulate(c, adj[s], seq, pf, record=(name == "DP_42d" and s == "half_edge"))
            rows.append({"policy": name, "eval": "boot_" + s, **summarise(res, T)})
            cdf[(name, s)] = [float(((res["done"] == 1) & (res["day"] <= m)).mean()) for m in range(1, T + 1)]
            if res["eq"] is not None:
                np.save(os.path.join(RES, "p06_sample_paths.npy"), res["eq"][:300])
                np.save(os.path.join(RES, "p06_sample_done.npy"), res["done"][:300])
    # constant-risk benchmarks under the same stop / take-profit machinery
    c = cfg(T=42); seq = boot_seq(NPATH, 42, np.arange(ND), 10, 11)
    for L in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        pf = policy_fn(c, const=L)
        st = test[: len(test) - 42]; hs = st[:, None] + np.arange(42)[None, :]
        rows.append({"policy": f"const_L{L}", "eval": "hist_test_2024H2_2026", **summarise(simulate(c, PATHS, hs, pf), 42)})
        for s in scen:
            rows.append({"policy": f"const_L{L}", "eval": "boot_" + s, **summarise(simulate(c, adj[s], seq, pf), 42)})
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "p06_pass_probabilities.csv"), index=False)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in cdf.items()}, open(os.path.join(RES, "p06_cdf.json"), "w"))
    c, dp = keep["DP_42d"]
    np.savez_compressed(os.path.join(RES, "p06_policy_DP_42d.npz"), EG=dp["EG"], pol1=dp["pol"][1], pol2=dp["pol"][2],
                        V1=dp["V"][1], V2=dp["V"][2], L=LG)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    cols = ["policy", "eval", "n", "p_pass", "p_fail_daily", "p_fail_total", "p_timeout", "p_phase1", "median_p1_days",
            "median_days_if_pass", "mean_L", "pass_by_10", "pass_by_21", "pass_by_30", "pass_by_42"]
    print(R[cols].round(3).to_string(index=False))
