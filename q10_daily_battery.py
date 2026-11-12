"""Daily-frequency strategies taken from the three literature folders, one engine, one protocol.

Every strategy uses the paper's own or textbook-default parameters, decides at the close of day t
with data <= t, earns day t+1, and pays 1 bp/side plus financing (see pf_lib).  Blocks:
  DESIGN 2005-2018 | VALID 2019 - 2021-11 | BROKER 2021-11 - 2026-10 (broker data, never used to select)
Paper index numbers refer to papers/NOTES.md.
"""
import os, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from hmmlearn.hmm import GaussianHMM
from pf_lib import *

warnings.filterwarnings("ignore")
D = load_daily()[ASSETS]
R0 = D.fillna(0.0)
LP = np.log1p(R0).cumsum()
VOL = ewvol(D)
COV = ewcov(D)
CLS = pd.Series(CLASS)
NCLS = CLS.map(CLS.value_counts())
Z0 = pd.DataFrame(0.0, index=D.index, columns=ASSETS)


def risk(sig: pd.DataFrame, cols=None) -> pd.DataFrame:
    """signal in [-1,1] per asset -> risk-unit weights, equal risk per asset class."""
    v = (sig / VOL).div(NCLS * 3, axis=1).reindex(columns=ASSETS).fillna(0.0)
    if cols is not None:
        v[[c for c in ASSETS if c not in cols]] = 0.0
    return v


STRATS: dict[str, pd.DataFrame] = {}


def add(name, sig, by_class=True):
    STRATS[name + "|ALL"] = risk(sig)
    if by_class:
        for k, cols in (("FX", FX), ("IDX", IDX), ("GOLD", GOLD)):
            STRATS[f"{name}|{k}"] = risk(sig, cols)


# ---- S1 cubic trend-reversion law (104, 105): score = b*phi + c*phi^3, T = 64/128/256 days
rhat = (D - D.expanding(250).mean()) / VOL.shift(1)
cub, lin = 0.0, 0.0
for T in (64, 128, 256):
    q = np.exp(-2.0 / T); M = np.sqrt(1 - np.exp(-4.0 / T))
    phi = rhat.fillna(0).ewm(alpha=1 - q, adjust=False).mean() * M / (1 - q)
    cub = cub + (1.29 * phi - 0.62 * phi ** 3) / 3; lin = lin + phi / 3
add("S1 cubic trend-reversion", (cub / 0.8).clip(-1, 1))
add("S1b linear trend (null)", (lin / 1.5).clip(-1, 1))

# ---- S2 Forecast-to-Fill style trend regime (118): standardised slope of EMA-smoothed log price
sm = LP.ewm(alpha=0.06, adjust=False).mean().diff()
zs = (sm - sm.rolling(2520, min_periods=500).mean()) / sm.rolling(2520, min_periods=500).std()
mom = np.sign(LP - LP.shift(60))
f2f = (zs.clip(-2, 2) / 2).where(np.sign(zs) == mom, 0.0)
add("S2 trend-momentum regime", f2f)
add("S2b same, long-only", f2f.clip(lower=0))
add("S2c Baz 3-speed trend", baz_trend(LP).clip(-1, 1))

# ---- S4 MACD(12,26,9) (21)
macd = LP.ewm(span=12, adjust=False).mean() - LP.ewm(span=26, adjust=False).mean()
add("S4 MACD(12,26,9)", np.sign(macd - macd.ewm(span=9, adjust=False).mean()))

# ---- S8 Hurst switch (90): momentum if H > 0.5 else mean reversion
def hurst(lp, win=250, lags=(2, 4, 8, 16)):
    s = np.log(pd.concat({k: lp.diff(k).rolling(win).std() for k in lags}, axis=1))
    x = np.log(np.array(lags)); x = x - x.mean()
    out = {}
    for a in lp.columns:
        y = s.xs(a, axis=1, level=1); out[a] = ((y - y.mean(axis=1).values[:, None]) * x).sum(axis=1) / (x ** 2).sum()
    return pd.DataFrame(out)
H = hurst(LP)
mo = np.sign(LP - LP.shift(20)); mr = -np.sign(LP - LP.rolling(5).mean())
add("S8 Hurst switch", mo.where(H > 0.5, mr))

# ---- S3 technical-rule universe (8, 9, 51)
rules = {}
for f in (1, 5, 10, 20):
    for s in (20, 50, 100, 200):
        if f < s: rules[f"ma_{f}_{s}"] = np.sign(LP.rolling(f).mean() - LP.rolling(s).mean())
for n in (10, 20, 50, 100, 200):
    hi, lo = LP.shift(1).rolling(n).max(), LP.shift(1).rolling(n).min()
    b = pd.DataFrame(np.where(LP > hi, 1.0, np.where(LP < lo, -1.0, np.nan)), index=LP.index, columns=ASSETS)
    rules[f"chan_{n}"] = b.ffill().fillna(0)
for n in (5, 10, 20, 60, 120, 250):
    rules[f"mom_{n}"] = np.sign(LP - LP.shift(n))
for n in (2, 5, 14):
    up = R0.clip(lower=0).rolling(n).mean(); dn = (-R0).clip(lower=0).rolling(n).mean(); rsi = 100 * up / (up + dn + 1e-12)
    for lo_, hi_ in ((20, 80), (30, 70)):
        rules[f"rsi_{n}_{lo_}"] = pd.DataFrame(np.where(rsi < lo_, 1.0, np.where(rsi > hi_, -1.0, 0.0)), index=LP.index, columns=ASSETS)
for n in (10, 20):
    zb = (LP - LP.rolling(n).mean()) / LP.rolling(n).std()
    for k in (1.5, 2.0):
        rules[f"boll_{n}_{k}"] = pd.DataFrame(np.where(zb < -k, 1.0, np.where(zb > k, -1.0, 0.0)), index=LP.index, columns=ASSETS)
RN = list(rules)
# per rule-asset net return per unit risk (cost on turnover included)
SR = (D / VOL.shift(1)).clip(-6, 6).fillna(0)
ra = {}
for k, sg in rules.items():
    sg = sg.fillna(0)
    ra[k] = sg.shift(1) * SR - sg.diff().abs().shift(1).fillna(0) * (COST_SIDE / VOL.shift(1)).clip(upper=0.1).fillna(0)
RA = pd.concat(ra, axis=1)                     # columns (rule, asset)
SG = pd.concat({k: v.fillna(0) for k, v in rules.items()}, axis=1)
print("rule universe:", len(RN), "rules x", len(ASSETS), "assets =", RA.shape[1], "rule-asset series")

# S3a: select with Benjamini-Hochberg FDR 10% on trailing 500 days, hold selection 126 days
sel_sig = Z0.copy(); nsel = []
for t0 in range(750, len(D), 126):
    win = RA.iloc[t0 - 500:t0]
    tstat = win.mean() / (win.std() + 1e-12) * np.sqrt(len(win))
    from scipy.stats import norm
    p = pd.Series(1 - norm.cdf(tstat.values), index=tstat.index).sort_values(); m = len(p)
    ok = p[p.values <= 0.10 * (np.arange(1, m + 1) / m)]
    keep = p.index[: len(ok)] if len(ok) else []
    nsel.append(len(keep))
    blk = slice(t0, min(t0 + 126, len(D)))
    if len(keep):
        s = SG.iloc[blk][list(keep)].T.groupby(level=1).mean().T
        sel_sig.iloc[blk] = s.reindex(columns=ASSETS).fillna(0).values
print("FDR-selected rule-assets per rebalance: mean %.1f, zero in %d of %d rebalances" % (np.mean(nsel), sum(n == 0 for n in nsel), len(nsel)))
add("S3a technical rules, FDR-selected", sel_sig, by_class=False)
# S3b: online expert aggregation, exponential weights on trailing 250-day performance
perf = RA.rolling(250, min_periods=120).sum().shift(1)
wts = np.exp((perf / np.sqrt(250)).clip(-5, 5)); wts = wts.div(wts.T.groupby(level=1).sum().T, level=1)
add("S3b technical rules, online learning", (SG * wts).T.groupby(level=1).sum().T.reindex(columns=ASSETS), by_class=True)
add("S3c technical rules, equal weight", SG.T.groupby(level=1).mean().T.reindex(columns=ASSETS), by_class=False)

# ---- S6 cointegration pairs with the standard rules (19, 23, 117, 121, 144, 145)
usd = pd.DataFrame({c: (-LP[p] if p in INV else LP[p]) for c, p in MAJORS.items()})
price = pd.concat([usd, LP[IDX], LP[GOLD]], axis=1)
legs = {**{c: (MAJORS[c], -1.0 if MAJORS[c] in INV else 1.0) for c in CCY}, **{a: (a, 1.0) for a in IDX + GOLD}}
cand = [(a, b) for grp in (CCY, IDX) for i, a in enumerate(grp) for b in grp[i + 1:]] + [("XAUUSD", c) for c in ("AUD", "CHF", "JPY")]
pw = Z0.copy(); npairs = []
for t0 in range(500, len(D), 125):
    form = price.iloc[t0 - 250:t0]; blk = slice(t0, min(t0 + 125, len(D))); cnt = 0
    acc = np.zeros((blk.stop - blk.start, len(ASSETS)))
    for a, b in cand:
        x, y = form[b].values, form[a].values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9: continue
        beta = np.cov(x, y)[0, 1] / np.var(x); sp = y - beta * x
        if adfuller(sp, maxlag=1, autolag=None)[1] > 0.05: continue
        mu, sd = sp.mean(), sp.std()
        z = ((price[a].values[blk] - beta * price[b].values[blk]) - mu) / sd
        pos = np.zeros(len(z)); cur = 0.0
        for i, zi in enumerate(z):
            if cur == 0 and abs(zi) > 2 and abs(zi) < 4: cur = -np.sign(zi)
            elif cur != 0 and (zi * cur > 0 or abs(zi) > 4): cur = 0.0
            pos[i] = cur
        va = VOL.iloc[t0 - 1]
        for leg, sgn in ((a, 1.0), (b, -beta)):
            asset, o = legs[leg]; j = ASSETS.index(asset)
            acc[:, j] += pos * sgn * o / (form[a].diff().std() + 1e-9)
        cnt += 1
    npairs.append(cnt)
    if cnt: pw.iloc[blk] = acc / cnt
print("cointegrated pairs traded per window: mean %.1f of %d candidates" % (np.mean(npairs), len(cand)))
STRATS["S6 cointegration pairs|ALL"] = pw

# ---- S7 / S11 long-only risk parity with and without a causal HMM volatility regime (95, 18, 50, 84, 85, 142)
lo = Z0.copy(); lo[IDX] = 1.0; lo[GOLD] = 1.0
rp = (lo / VOL); rp[IDX] = rp[IDX] / len(IDX) / 2; rp[GOLD] = rp[GOLD] / 2
STRATS["S11 long-only stock+gold, vol-controlled|ALL"] = rp
x = R0["SP500"].values.reshape(-1, 1) * 100
expo = np.full(len(D), np.nan); state_hi = np.full(len(D), np.nan)
for t0 in range(1000, len(D), 250):
    m = GaussianHMM(3, covariance_type="diag", n_iter=50, random_state=0).fit(x[:t0])
    order = np.argsort(m.covars_.reshape(3)); A = m.transmat_; mu = m.means_.reshape(3); sd = np.sqrt(m.covars_.reshape(3))
    a = m.startprob_.copy()
    for t in range(max(t0 - 250, 0), min(t0 + 250, len(D))):          # forward filter only (no smoothing)
        lik = np.exp(-0.5 * ((x[t, 0] - mu) / sd) ** 2) / sd
        a = (a @ A) * lik; a = a / a.sum()
        if t >= t0:
            expo[t] = a[order[0]] * 1.0 + a[order[1]] * 0.5 + a[order[2]] * 0.0
expo = pd.Series(expo, index=D.index).fillna(1.0)
STRATS["S7 same with HMM volatility-regime exposure|ALL"] = rp.mul(expo, axis=0)

# ---- S12 gold turn-of-year (149): long January, short July
g = Z0.copy(); g.loc[D.index.month == 1, "XAUUSD"] = 1.0; g.loc[D.index.month == 7, "XAUUSD"] = -1.0
STRATS["S12 gold January long / July short|GOLD"] = g / VOL
g2 = Z0.copy(); g2.loc[D.index.month == 1, "XAUUSD"] = 1.0
STRATS["S12b gold January long only|GOLD"] = g2 / VOL

# ---- reference sleeves from the first study
from p05_book import rl1
STRATS["REF index dip-buy 1-day (92, 102)|IDX"] = rl1(D, VOL)
STRATS["REF gold long-only Baz trend|GOLD"] = raw_sleeves(D)["GL"]

# ---- S5 carry (2, 22, 39): needs its own P&L because the financing IS the strategy
rt = pd.read_csv(os.path.join(Q, "forexStrategyDesign", "data", "rates", "rates_3M_wide.csv"), parse_dates=[0], index_col=0)
rt = rt[["USD"] + CCY].ffill().shift(1).reindex(D.index, method="ffill")      # lag one month
diff = rt[CCY].sub(rt["USD"], axis=0) / 100.0
rank = diff.rank(axis=1)
csig = pd.DataFrame(np.where(rank >= 6, 1.0, np.where(rank <= 2, -1.0, 0.0)), index=D.index, columns=CCY)
cw = Z0.copy()
for c, p in MAJORS.items():
    cw[p] = (-csig[c] if p in INV else csig[c]) / VOL[p] / 4
STRATS["S5 FX carry, top-2 vs bottom-2|FX"] = cw

# ---------------------------------------------------------------- evaluate
periods = {"DESIGN": ("2005-01-01", DESIGN_END), "VALID": (DESIGN_END + pd.Timedelta(days=1), VALID_END),
           "BROKER": (BROKER_START, "2026-10-30")}
rows, rets = [], {}
for name, v in STRATS.items():
    w = unit_vol(v, COV)
    if name.startswith("S5"):
        net, cost, _ = pnl(w, D, financing=False)
        wl = w.shift(1).fillna(0)
        carry = sum((wl[p] * (-diff[c] if p in INV else diff[c])) for c, p in MAJORS.items()) / ANN
        net = net + carry - wl.abs().sum(axis=1) * 0.01 / ANN           # earn the differential, pay 1%/yr markup
    else:
        net, cost, _ = pnl(w, D)
    gross, _, _ = pnl(w, D, 0.0, False)
    rets[name] = net
    r = {"strategy": name.split("|")[0], "universe": name.split("|")[1]}
    for pn, (a, b) in periods.items():
        r[pn] = sharpe(net[a:b]); r[pn + "_gross"] = sharpe(gross[a:b])
    r["turnover"] = float(w["2005":].diff().abs().sum(axis=1).mean())
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(RES, "q10_daily_battery.csv"), index=False)
pd.DataFrame(rets).to_parquet(os.path.join(RES, "q10_daily_returns.parquet"))
pd.to_pickle({k: unit_vol(v, COV) for k, v in STRATS.items()}, os.path.join(RES, "q10_weights.pkl"))
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(T.round(2).to_string(index=False))
