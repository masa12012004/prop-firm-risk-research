"""Month-by-month results of every Phase-5 gold strategy that can be run on hourly/daily bars,
plus a "learn from every closed trade" layer on the regime x cubic hybrid.

Strategy code is imported unchanged from gold_hist_Data/phase5_literature (or copied verbatim where it
lives under __main__ there).  Data: MT5 XAUUSD H1, true hourly bars from 2019.  Models are refitted at
the start of every month from 2019-07 on, always on earlier data only.  Costs: 0.85 bp half-spread +
0.2 bp slippage per unit traded (the Phase-5 convention); no overnight financing is charged.

Learning layer (applies to the hybrid's trades; a trade = one uninterrupted long or short episode):
  before each new trade, a model that has seen ONLY trades already closed gives P(win); the trade is
  sized  m = clip(1 + (P - base) / 0.1, 0, 2), base = win rate of the trades closed so far.  The model is updated after every closed trade.
    online logistic regression (updated trade by trade) | LightGBM (refit every 20 closed trades)
    GRU over the last 16 closed trades (retrained every 40 trades) | average of the three
    naive rule: 1.5x after a winning run of 10 trades, 0.5x after a losing one
"""
import os, sys, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
from sklearn.linear_model import SGDClassifier
warnings.filterwarnings("ignore"); torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
P5 = os.path.join(os.path.dirname(HERE), "gold_hist_Data", "phase5_literature"); sys.path.insert(0, P5)
import common, s01_f2f, s02_cubic_tsmom as s02, s05_hurst_q as s05
from pf_lib import Q, RES, FIG, sharpe

START, END = pd.Timestamp("2019-07-01", tz="US/Eastern"), pd.Timestamp("2026-09-18", tz="US/Eastern")
REPORT = "2023-10-01"
common.OOS_START, common.OOS_END = START, END
def refits(index, start=START, end=END):
    idx = index[(index >= start) & (index < end)]
    return [pd.Timestamp(x) for x in pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first().tolist()]
s01_f2f.monthly_refit_points = refits; s02.monthly_refit_points = refits

g = pd.read_csv(os.path.join(Q, "XAUUSDstrat02", "ohlc_xauusd", "XAUUSD_H1.csv"), parse_dates=["time"])
g = g[g.time >= "2018-12-01"]
CUT, TAG = os.environ.get("Q17_CUT"), os.environ.get("Q17_TAG", "")      # audit: rerun on data truncated at CUT
if CUT: g = g[g.time < CUT]
g["dt"] = (g["time"] - pd.Timedelta(hours=7)).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
g = g.dropna(subset=["dt"]).drop_duplicates("dt").set_index("dt").sort_index()
h1 = g[["open", "high", "low", "close"]].copy(); h1["mid"] = h1["close"]
h1["ret"] = h1["mid"].pct_change().fillna(0.0); h1["lret"] = np.log(h1["mid"]).diff().fillna(0.0)
h1["hs"] = 0.85e-4; h1["tday"] = (h1.index + pd.Timedelta(hours=7)).normalize().tz_localize(None)
gb = h1.groupby("tday")
dly = pd.DataFrame({"open": gb["open"].first(), "high": gb["high"].max(), "low": gb["low"].min(), "close": gb["close"].last(), "n": gb.size()})
dly = dly[dly.n > 12]; dly["hs"] = 0.85e-4
dly.index = pd.DatetimeIndex(dly.index, name="dt").tz_localize("US/Eastern") + pd.Timedelta(hours=16, minutes=55)
dly["mid"] = dly["close"]; dly["ret"] = dly["mid"].pct_change().fillna(0.0); dly["lret"] = np.log(dly["mid"]).diff().fillna(0.0)
dly["tday"] = dly.index.tz_localize(None).normalize()
bpy = common.bars_per_year(h1); print("H1 bars", len(h1), "| daily bars", len(dly), flush=True)

# ---------------------------------------------------------------- base strategies (Phase-5 logic)
fc, _ = s02.cubic_positions(h1); sig = fc.clip(-1, 1)
cubic = common.vol_target(sig, h1["ret"], bpy, 0.15, span=24 * 20)
cubic_strong = common.vol_target(sig.where(sig.abs() > 0.5, 0.0), h1["ret"], bpy, 0.15, span=24 * 20)
tsmom, _ = s02.tsmom_positions(h1, bpy)
f2f_ls, _ = s01_f2f.build(dly, True); f2f_lo, _ = s01_f2f.build(dly, False)
key = pd.DataFrame({"t": (h1.index + pd.Timedelta(hours=1)).tz_convert("UTC")}, index=h1.index).reset_index()
dk = pd.DataFrame({"t": (dly.index + pd.Timedelta(minutes=5)).tz_convert("UTC"), "f2f": f2f_ls}).sort_values("t")
regime = pd.merge_asof(key.sort_values("t"), dk, on="t").set_index("dt")["f2f"].reindex(h1.index).fillna(0.0)
hyb = cubic.where((np.sign(cubic) == np.sign(regime)) & (regime != 0), 0.0)
# Hurst switch and tabular Q-learning: verbatim from s05_hurst_q.__main__
Hh = s05.rolling_hurst(h1["lret"].values); mom, mr, diff = s05.base_signals(h1)
nd = diff / common.atr(h1, 14).values; pos_sw = np.zeros(len(h1)); pos_q = np.zeros(len(h1))
fwd = h1["ret"].shift(-1).fillna(0).values; cost = h1["hs"].values + 0.2e-4; rf = refits(h1.index)
for i, r0 in enumerate(rf):
    r1 = rf[i + 1] if i + 1 < len(rf) else h1.index[-1] + pd.Timedelta(days=1)
    tr = np.asarray((h1.index < r0) & np.isfinite(Hh)); te = np.asarray((h1.index >= r0) & (h1.index < r1)); best = None
    for b in np.round(np.arange(0.40, 0.71, 0.025), 3):
        s = s05.pnl(h1, np.where(Hh > b, mom, mr), tr)
        if best is None or s > best[0]: best = (s, b)
    pos_sw[te] = np.where(Hh[te] > best[1], mom[te], mr[te])
    q1 = np.quantile(nd[tr & np.isfinite(nd)], [0.2, 0.4, 0.6, 0.8]); s1 = np.searchsorted(q1, nd); s0 = np.r_[2, s1[:-1]]
    state = s1 * 10 + s0 * 2 + (Hh > 0.5).astype(int); Rq = {}
    for act in (-1, 1):
        Rq[act] = pd.DataFrame({"s": state[tr], "r": (act * fwd - cost)[tr]}).groupby("s")["r"].agg(["mean", "std", "count"])
    amap = {}
    for s in np.unique(state[tr]):
        ba, bt_ = 0, 2.0
        for act in (-1, 1):
            if s in Rq[act].index:
                m_, sd_, n_ = Rq[act].loc[s]; t_ = m_ / (sd_ / np.sqrt(n_)) if sd_ > 0 and n_ > 30 else 0
                if t_ > bt_: ba, bt_ = act, t_
        amap[s] = ba
    pos_q[te] = pd.Series(state[te]).map(amap).fillna(0).values
h5 = common.vol_target(pd.Series(1.0, index=dly.index), dly["ret"], common.ANN, 0.15, span=20, cap=1.5)

def bt(df, pos): return common.backtest(df, pos, "x", save=False)
def daily(b, shift=False):
    r = b.groupby("tday")["net"].sum(); return r.shift(1).fillna(0.0) if shift else r
BT = {"Hybrid: regime x cubic": bt(h1, hyb)}
R = pd.DataFrame({"Buy and hold gold": daily(bt(dly, np.ones(len(dly))), True),
                  "F2F regime long-only": daily(bt(dly, f2f_lo), True), "F2F regime long/short": daily(bt(dly, f2f_ls), True),
                  "Cubic trend-reversion": daily(bt(h1, cubic)), "Cubic, strong signals only": daily(bt(h1, cubic_strong)),
                  "TSMOM": daily(bt(h1, tsmom)), "SMA momentum": daily(bt(h1, mom)), "Bollinger mean reversion": daily(bt(h1, mr)),
                  "Hurst switch": daily(bt(h1, pos_sw)), "Q-learning (tabular)": daily(bt(h1, pos_q)),
                  "Vol-controlled long gold": daily(bt(dly, h5), True), "Hybrid: regime x cubic": daily(BT["Hybrid: regime x cubic"])}).fillna(0.0)
base = [c for c in R if c not in ("Buy and hold gold", "Vol-controlled long gold", "Hybrid: regime x cubic")]
vol60 = R[base].rolling(60, min_periods=20).std().shift(1); weq = (1 / vol60).replace([np.inf, -np.inf], np.nan); weq = weq.div(weq.sum(1), axis=0).fillna(0.0)
def vol_control(r, tgt=0.10, win=20, cap=2.0):
    return r * (tgt / (r.rolling(win, min_periods=10).std().shift(1) * np.sqrt(252))).clip(upper=cap).fillna(0.0)
R["Ensemble: equal risk, all base"] = vol_control((weq * R[base]).sum(1))
W = pd.DataFrame(0.0, index=R.index, columns=base); month = pd.Series(R.index.to_period("M"), index=R.index)
for m in month.unique():
    days = R.index[month == m]; hist = R[base][R.index < days[0]]
    if len(hist) < 60: w = weq.loc[days[0]]
    else:
        w = ((hist.mean() / hist.std().replace(0, np.nan)).clip(lower=0) / hist.std()).fillna(0.0); w = w / w.sum() if w.sum() > 0 else weq.loc[days[0]]
    W.loc[days] = w.values
R["Ensemble: walk-forward Sharpe-weighted"] = vol_control((W * R[base]).sum(1))

# ---------------------------------------------------------------- learn from every closed trade (hybrid)
p = hyb.values; sgn = np.sign(p); n = len(p)
chg = np.flatnonzero(np.diff(np.r_[0, sgn]) != 0)
trades = [(a, (chg[k + 1] if k + 1 < len(chg) else n)) for k, a in enumerate(chg) if sgn[a] != 0]
net = (pd.Series(p, index=h1.index) * h1["ret"].shift(-1).fillna(0) - pd.Series(p, index=h1.index).diff().abs().fillna(0) * (h1["hs"] + 0.2e-4)).values
tret = np.array([net[a:b].sum() for a, b in trades]); win = (tret > 0).astype(int)
lp = np.log(h1["mid"]).values; hv = h1["ret"].ewm(span=480, min_periods=50).std().shift(1).bfill().values
v24 = h1["ret"].ewm(span=24).std().values; ma20 = h1["mid"].rolling(20).mean().values; sd20 = h1["mid"].rolling(20).std().values
hours = h1.index.hour.values; fcv = fc.values; regv = regime.values
def feats(k):
    a, _ = trades[k]; s = sgn[a]; closed = [j for j in range(max(0, k - 60), k) if trades[j][1] <= a]
    last = tret[closed[-10:]] if closed else np.array([0.0])
    f = [s, abs(fcv[a]), abs(regv[a]), np.sin(2 * np.pi * hours[a] / 24), np.cos(2 * np.pi * hours[a] / 24), v24[a] / (hv[a] + 1e-12),
         s * (lp[a] - lp[max(a - 24, 0)]) / (hv[a] * 24 ** .5 + 1e-12), s * (lp[a] - lp[max(a - 120, 0)]) / (hv[a] * 120 ** .5 + 1e-12),
         s * (lp[a] - lp[max(a - 480, 0)]) / (hv[a] * 480 ** .5 + 1e-12), np.nan_to_num(Hh[a], nan=0.5),
         s * np.nan_to_num((h1["mid"].values[a] - ma20[a]) / (sd20[a] + 1e-12)),
         float(np.sign(last[-1])), float((last > 0).mean()), float(last.mean() / (np.abs(last).mean() + 1e-12)),
         min((a - trades[closed[-1]][1]) if closed else 100, 200) / 100.0]
    return np.nan_to_num(np.array(f, dtype=np.float32))
X = np.array([feats(k) for k in range(len(trades))]); NT = len(trades)
entry_t = h1.index[[a for a, _ in trades]]
print(f"hybrid trades: {NT}, win rate {win.mean():.3f}, mean trade {tret.mean()*1e4:.2f} bp, median length {np.median([b-a for a,b in trades]):.0f} h", flush=True)

class G(nn.Module):
    def __init__(s, d):
        super().__init__(); s.rnn = nn.GRU(d + 1, 16, batch_first=True); s.head = nn.Sequential(nn.Linear(16 + d, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(s, seq, cur): return s.head(torch.cat([s.rnn(seq)[0][:, -1], cur], 1)).squeeze(1)
SEQ = 16
def seq_of(k, closed_idx, mu, sd):
    idx = closed_idx[-SEQ:]; z = np.zeros((SEQ, X.shape[1] + 1), dtype=np.float32)
    if len(idx): z[-len(idx):] = np.c_[(X[idx] - mu) / sd, np.sign(tret[idx])]
    return z

P = {m: np.full(NT, np.nan) for m in ("Online logistic", "LightGBM", "GRU", "Average of three")}; naive = np.ones(NT); BASE = np.full(NT, np.nan)
sgd = SGDClassifier(loss="log_loss", alpha=1e-3, learning_rate="constant", eta0=0.01, random_state=0); sgd_ready = False
gbm = None; gru = None; closed = []; pending = []; n_at_gbm = n_at_gru = 0; WARM = 100
order_exit = sorted(range(NT), key=lambda k: trades[k][1])
for k in range(NT):
    a = trades[k][0]
    newly = [j for j in pending if trades[j][1] <= a]; pending = [j for j in pending if trades[j][1] > a]
    for j in newly:                                             # learn from each trade as it closes
        closed.append(j)
        mu, sd = X[closed].mean(0), X[closed].std(0) + 1e-6
        sgd.partial_fit(((X[j] - mu) / sd)[None], [win[j]], classes=[0, 1]); sgd_ready = True
    nc = len(closed)
    if nc >= WARM:
        mu, sd = X[closed].mean(0), X[closed].std(0) + 1e-6; xs = ((X[k] - mu) / sd)[None]; BASE[k] = win[closed].mean()
        P["Online logistic"][k] = sgd.predict_proba(xs)[0, 1]
        if gbm is None or nc - n_at_gbm >= 20:
            gbm = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.05, num_leaves=4, min_child_samples=25, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=1).fit(X[closed], win[closed]); n_at_gbm = nc
        P["LightGBM"][k] = gbm.predict_proba(X[k][None])[0, 1]
        if gru is None or nc - n_at_gru >= 40:
            S = np.array([seq_of(j, closed[:i], mu, sd) for i, j in enumerate(closed)]); C = (X[closed] - mu) / sd
            gru = G(X.shape[1]); opt = torch.optim.Adam(gru.parameters(), 3e-3, weight_decay=1e-3)
            St, Ct, yt = torch.tensor(S), torch.tensor(C.astype(np.float32)), torch.tensor(win[closed].astype(np.float32))
            for ep in range(25):
                opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(gru(St, Ct), yt).backward(); opt.step()
            n_at_gru = nc; gru.eval()
        with torch.no_grad():
            P["GRU"][k] = torch.sigmoid(gru(torch.tensor(seq_of(k, closed, mu, sd)[None]), torch.tensor(xs.astype(np.float32)))).item()
        P["Average of three"][k] = np.mean([P[m][k] for m in ("Online logistic", "LightGBM", "GRU")])
        l10 = tret[closed[-10:]]; naive[k] = 1.5 if l10.mean() > 0 else 0.5
    pending.append(k)
from sklearn.metrics import roc_auc_score
diag = []; POS = {"Hybrid": p.copy(), "ret": h1["ret"].values, "regime": regv, "cubic": cubic.values}
for m, pr in P.items():
    ok = np.isfinite(pr); inrep = ok & (entry_t >= pd.Timestamp(REPORT, tz="US/Eastern"))
    diag.append({"learner": m, "trades scored": int(ok.sum()), "AUC all": roc_auc_score(win[ok], pr[ok]), "AUC last 3y": roc_auc_score(win[inrep], pr[inrep]),
                 "win rate, P > base+5pt": win[ok & (pr > BASE + 0.05)].mean(), "mean trade bp, P > base+5pt": tret[ok & (pr > BASE + 0.05)].mean() * 1e4,
                 "win rate, P < base-5pt": win[ok & (pr < BASE - 0.05)].mean(), "mean trade bp, P < base-5pt": tret[ok & (pr < BASE - 0.05)].mean() * 1e4,
                 "mean size": float(np.nanmean(np.clip(1 + (pr - BASE) / 0.1, 0, 2)))})
    mult = np.where(ok, np.clip(1 + (pr - BASE) / 0.1, 0, 2), 1.0); pos2 = p.copy()
    for (a, b), mm in zip(trades, mult): pos2[a:b] *= mm
    R[f"Hybrid + learning: {m}"] = daily(bt(h1, pos2)).reindex(R.index).fillna(0.0); POS[m] = pos2.copy()
pos2 = p.copy()
for (a, b), mm in zip(trades, naive): pos2[a:b] *= mm
R["Hybrid + naive last-10-trades sizing"] = daily(bt(h1, pos2)).reindex(R.index).fillna(0.0)
print(pd.DataFrame(diag).round(3).to_string(index=False))
pd.DataFrame(POS, index=h1.index).to_parquet(os.path.join(RES, f"q17_positions{TAG}.parquet"))
if TAG: sys.exit(0)                                             # audit run: positions only, leave the main outputs alone

# ---------------------------------------------------------------- monthly tables
R.to_parquet(os.path.join(RES, "q17_daily_returns.parquet"))
Rr = R[REPORT:]; M = ((1 + Rr).groupby(Rr.index.to_period("M")).prod() - 1) * 100
M.round(2).to_csv(os.path.join(RES, "q17_monthly_returns_pct.csv"))
summ = pd.DataFrame({"total %": ((1 + Rr).prod() - 1) * 100, "Sharpe": Rr.apply(sharpe), "months up": (M > 0).sum(), "months down": (M < 0).sum(),
                     "best month %": M.max(), "worst month %": M.min(), "max drawdown %": ((1 + Rr).cumprod() / (1 + Rr).cumprod().cummax() - 1).min() * 100,
                     "Sharpe 2020-07 to 2023-09": R["2020-07-01":"2023-09-30"].apply(sharpe)})
summ.round(2).to_csv(os.path.join(RES, "q17_summary.csv"))
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100); pd.set_option("display.max_columns", 30)
print(summ.round(2).to_string())
show = ["Buy and hold gold", "Hybrid: regime x cubic", "Hybrid + learning: Average of three", "F2F regime long-only", "Cubic trend-reversion", "TSMOM", "Hurst switch", "Ensemble: walk-forward Sharpe-weighted"]
print(M[show].round(1).to_string())
