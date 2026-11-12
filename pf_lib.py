"""Shared library: universe, data loading, signals, portfolio construction, stats.

Conventions
  * A trading day ends at 21:00 UTC (~17:00 New York, the FX roll).
  * Weights w[t] are decided with information up to the close of day t and earn
    the return of day t+1.  Nothing in this file looks forward.
  * Weights are notional / equity.  "Unit" portfolios target 1% ex-ante daily vol.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(ROOT)
DATA, RES, FIG = (os.path.join(ROOT, d) for d in ("data", "results", "figs"))

CCY = ["EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY"]
MAJORS = {"EUR": "EURUSD", "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
          "CAD": "USDCAD", "CHF": "USDCHF", "JPY": "USDJPY"}
FX = list(MAJORS.values())
INV = {"USDCAD", "USDCHF", "USDJPY"}            # USD is the base currency
IDX = ["SP500", "NAS100", "DOW30", "DAX40", "FTSE100", "JPN225"]
GOLD = ["XAUUSD"]
ASSETS = FX + IDX + GOLD
CLASS = {**{a: "FX" for a in FX}, **{a: "IDX" for a in IDX}, "XAUUSD": "GOLD"}

# --- costs (fractions of notional) -------------------------------------------
COST_SIDE = 1.0e-4            # half-spread + commission + slippage, per side
FIN_LONG = {"IDX": 0.06, "GOLD": 0.06, "FX": 0.01}   # annual financing on longs
FIN_SHORT = {"IDX": 0.00, "GOLD": 0.00, "FX": 0.01}
ANN = 252
BROKER_START = pd.Timestamp("2021-11-08")
DESIGN_END = pd.Timestamp("2018-12-31")        # signals/sleeve choice may look at <= this
VALID_END = pd.Timestamp("2021-11-05")         # frozen validation block


# --- data ---------------------------------------------------------------------
def load_daily() -> pd.DataFrame:
    """Daily simple returns, long history (Yahoo before the broker window)."""
    return pd.read_parquet(os.path.join(DATA, "daily_ret.parquet"))


def load_hourly() -> pd.DataFrame:
    """Hourly simple returns on a common UTC grid, broker window only."""
    return pd.read_parquet(os.path.join(DATA, "hourly_ret.parquet"))


def day_key(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bar OPEN time (UTC) -> trading day that ends 21:00 UTC."""
    return (idx + pd.Timedelta(hours=3)).normalize()


# --- signals --------------------------------------------------------------------
def ewvol(r: pd.DataFrame, span: int = 30) -> pd.DataFrame:
    return np.sqrt((r ** 2).ewm(span=span, min_periods=20).mean())


def baz_trend(logp: pd.DataFrame) -> pd.DataFrame:
    """Baz et al. (2015) CTA trend signal: three EWMA crossovers, vol-normalised,
    passed through the response u = z*exp(-z^2/4)/0.89 which fades extreme trends
    (the same shape as the Schmidhuber cubic trend-reversion law)."""
    out = 0.0
    dv = logp.diff().rolling(63, min_periods=40).std()
    for s, l in ((8, 24), (16, 48), (32, 96)):
        x = logp.ewm(halflife=np.log(0.5) / np.log(1 - 1 / s)).mean() - \
            logp.ewm(halflife=np.log(0.5) / np.log(1 - 1 / l)).mean()
        y = x / dv
        z = y / y.rolling(252, min_periods=120).std()
        out = out + z * np.exp(-z ** 2 / 4) / 0.89
    return out / 3.0


def fx_cross_panel(r: pd.DataFrame) -> pd.DataFrame:
    """Log USD-value of each currency -> log prices of all 28 G8 pairs."""
    usd = {}
    for c, p in MAJORS.items():
        lp = np.log1p(r[p]).cumsum()
        usd[c] = -lp if p in INV else lp
    usd["USD"] = usd["EUR"] * 0.0
    names = CCY + ["USD"]
    cols = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cols[(a, b)] = usd[a] - usd[b]
    return pd.DataFrame(cols)


def net_fx(sig_pairs: pd.DataFrame, vol_pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-pair risk-scaled signals -> net weight on the 7 USD majors."""
    cur = {c: 0.0 for c in CCY + ["USD"]}
    v = (sig_pairs / vol_pairs) / sig_pairs.shape[1]
    for (a, b) in sig_pairs.columns:
        cur[a] = cur[a] + v[(a, b)]
        cur[b] = cur[b] - v[(a, b)]
    w = {}
    for c, p in MAJORS.items():
        w[p] = -cur[c] if p in INV else cur[c]
    return pd.DataFrame(w)


def raw_sleeves(r: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Un-normalised sleeve weights (risk units) for every tradable asset."""
    r = r[ASSETS]
    logp = np.log1p(r.fillna(0.0)).cumsum()
    vol = ewvol(r)
    z = pd.DataFrame(0.0, index=r.index, columns=ASSETS)

    # T: diversified trend. Equal risk to FX / indices / gold.
    u = baz_trend(logp[IDX + GOLD])
    cp = fx_cross_panel(r)
    wfx = net_fx(baz_trend(cp), ewvol(cp.diff()))
    T = z.copy()
    T[FX] = wfx
    T[IDX] = (u[IDX] / vol[IDX]) / len(IDX)
    T[GOLD] = u[GOLD] / vol[GOLD]

    # B: equity risk premium, always long, equal risk.
    B = z.copy()
    B[IDX] = (1.0 / vol[IDX]) / len(IDX)

    # R: index short-term reversal (buy 5-day losers, sell 5-day winners).
    z5 = (r[IDX].rolling(5).sum() / (vol[IDX] * np.sqrt(5))).clip(-2, 2) / 2
    R = z.copy()
    R[IDX] = (-z5 / vol[IDX]) / len(IDX)
    # RL: long-only dip buying variant.
    RL = z.copy()
    RL[IDX] = ((-z5).clip(lower=0) / vol[IDX]) / len(IDX)

    # GL: gold long-only trend (Forecast-to-Fill style regime).
    GL = z.copy()
    GL[GOLD] = u[GOLD].clip(lower=0) / vol[GOLD]
    return {"T": T, "B": B, "R": R, "RL": RL, "GL": GL}


def ewcov(r: pd.DataFrame, span: int = 60) -> np.ndarray:
    """EWMA covariance per day, shape (T, N, N). Zero-mean."""
    x = r.fillna(0.0).to_numpy()
    a = 2.0 / (span + 1)
    out = np.empty((len(x), x.shape[1], x.shape[1]))
    c = np.cov(x[:60].T) if len(x) > 60 else np.eye(x.shape[1]) * 1e-4
    for t in range(len(x)):
        c = (1 - a) * c + a * np.outer(x[t], x[t])
        out[t] = c
    return out


def unit_vol(v: pd.DataFrame, cov: np.ndarray, target: float = 0.01) -> pd.DataFrame:
    """Scale raw weights to `target` ex-ante daily vol; ex-ante vol is floored at
    half its expanding median so weak-signal days are not levered into noise."""
    x = v.fillna(0.0).to_numpy()
    s = np.sqrt(np.einsum("ti,tij,tj->t", x, cov, x))
    s = pd.Series(s, index=v.index).replace(0.0, np.nan)
    floor = 0.5 * s.expanding(min_periods=250).median()
    s = np.maximum(s, floor.fillna(s))
    return v.mul(target / s, axis=0).fillna(0.0)


def pnl(w: pd.DataFrame, r: pd.DataFrame, cost_mult: float = 1.0, financing: bool = True):
    """Net daily return of weights w (decided at t, earning t+1) + cost series."""
    r = r[w.columns].fillna(0.0)
    wl = w.shift(1).fillna(0.0)
    gross = (wl * r).sum(axis=1)
    cost = (w.diff().abs().shift(1).fillna(0.0)).sum(axis=1) * COST_SIDE * cost_mult
    fin = 0.0
    if financing:
        for a in w.columns:
            k = CLASS[a]
            fin = fin + (wl[a].clip(lower=0) * FIN_LONG[k] + (-wl[a]).clip(lower=0) * FIN_SHORT[k]) / 365.0
        fin = fin * 365.0 / ANN          # charge calendar days on trading days
    return gross - cost - fin, cost, fin


# --- statistics -------------------------------------------------------------------
def sharpe(x) -> float:
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(ANN)) if s > 0 else 0.0


def block_boot_idx(n: int, length: int, block: int, rng) -> np.ndarray:
    """Circular moving-block bootstrap indices."""
    k = int(np.ceil(length / block))
    st = rng.integers(0, n, size=k)
    return ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:length]


def sharpe_ci(x, block: int = 21, nb: int = 2000, seed: int = 7):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    s = [sharpe(x[block_boot_idx(len(x), len(x), block, rng)]) for _ in range(nb)]
    return float(np.percentile(s, 5)), float(np.percentile(s, 95))


def stats(x: pd.Series, ci: bool = True) -> dict:
    x = x.dropna()
    eq = (1 + x).cumprod()
    d = {"days": len(x), "ann_ret": x.mean() * ANN, "ann_vol": x.std() * np.sqrt(ANN),
         "sharpe": sharpe(x), "max_dd": float((eq / eq.cummax() - 1).min()),
         "worst_day": float(x.min()), "skew": float(x.skew())}
    if ci and len(x) > 250:
        d["sh_lo"], d["sh_hi"] = sharpe_ci(x.to_numpy())
    return d
