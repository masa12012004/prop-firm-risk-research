"""Deep-learning and reinforcement-learning models from the papers that p04_ml.py did not cover.

  BiLSTM + attention      (38, 113)        LSTM-ARIMA hybrid        (81)
  Direction classifier    (34, 148 TATS)   Policy-gradient LSTM maximising net Sharpe (115, 143, 36)
  Double DQN, 3 actions, position in the state (7, 58, 15, 116)
Same protocol as p04: fit A (train <= 2015, early-stop 2016-2018) -> VALID; fit B (train <= 2019,
early-stop 2020 - 2021-10) -> BROKER.  Costs and financing as everywhere else.
"""
import os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
from scipy.stats import spearmanr
from pf_lib import *
warnings.filterwarnings("ignore"); torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
SEQ = 30
D = load_daily()[ASSETS]; cov = ewcov(D); vol = ewvol(D)
sr = (D / vol.shift(1)).clip(-5, 5); y = sr.shift(-1)
usd = sr[FX].mul([-1 if p not in INV else 1 for p in FX], axis=1).mean(axis=1)
mkt = sr[["SP500", "NAS100", "DOW30"]].mean(axis=1)
vr = (D.rolling(5).std() / D.rolling(60).std()).clip(0, 5)
rho = sr.rolling(250).apply(lambda x: np.corrcoef(x[1:], x[:-1])[0, 1], raw=True) if False else sr.rolling(250).corr(sr.shift(1))
ar_res = sr - rho.shift(1) * sr.shift(1)                       # AR(1) residual, coefficient from the past only
trend = baz_trend(np.log1p(D.fillna(0)).cumsum())
CH = {a: np.nan_to_num(np.stack([sr[a], mkt, usd, sr["XAUUSD"], vr[a], ar_res[a]], axis=1)).astype(np.float32) for a in ASSETS}
TAB = {a: np.nan_to_num(np.stack([sr[a], sr[a].rolling(5).sum() / 5 ** .5, sr[a].rolling(21).sum() / 21 ** .5,
                                  sr[a].rolling(63).sum() / 63 ** .5, trend[a], vr[a], mkt, usd, sr["XAUUSD"],
                                  float(CLASS[a] == "FX") + 0 * mkt, float(CLASS[a] == "IDX") + 0 * mkt], axis=1)).astype(np.float32) for a in ASSETS}
Y = {a: np.nan_to_num(y[a].values).astype(np.float32) for a in ASSETS}
dates = D.index; ix = lambda d: int(dates.searchsorted(pd.Timestamp(d)))
FITS = {"VALID": ("2005-06-01", "2015-12-24", "2016-01-01", "2018-12-24", "2019-01-01", str(VALID_END.date())),
        "BROKER": ("2005-06-01", "2019-12-24", "2020-01-01", "2021-10-29", str(BROKER_START.date()), "2026-10-29")}
cls_n = pd.Series({a: (3 * sum(CLASS[b] == CLASS[a] for b in ASSETS)) for a in ASSETS})


def samples(a0, a1, nch):
    X, S, T, Yv, key = [], [], [], [], []
    for a in ASSETS:
        for j in range(max(ix(a0), 300), ix(a1) + 1):
            if j >= len(dates) - 1: break
            X.append(CH[a][j - SEQ + 1:j + 1, :nch]); T.append(TAB[a][j]); Yv.append(Y[a][j]); key.append((a, j))
    return torch.tensor(np.array(X)), torch.tensor(np.array(T)), torch.tensor(np.array(Yv)), key


class BiLSTMAtt(nn.Module):
    def __init__(s, nch):
        super().__init__(); s.rnn = nn.LSTM(nch, 16, batch_first=True, bidirectional=True); s.att = nn.Linear(32, 1); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x):
        h = s.rnn(x)[0]; w = torch.softmax(s.att(h), 1); return s.head((h * w).sum(1)).squeeze(1)


class LSTMNet(nn.Module):
    def __init__(s, nch):
        super().__init__(); s.rnn = nn.LSTM(nch, 32, batch_first=True); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x): return s.head(s.rnn(x)[0][:, -1]).squeeze(1)


def fit_sup(net_cls, nch, f):
    Xt, _, yt, _ = samples(f[0], f[1], nch); Xv, _, yv, _ = samples(f[2], f[3], nch); Xe, _, _, key = samples(f[4], f[5], nch)
    m = net_cls(nch); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3); best, st, bad = -9, None, 0
    for ep in range(10):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]; opt.zero_grad(); nn.functional.mse_loss(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): ic = spearmanr(m(Xv).numpy(), yv.numpy())[0]
        if ic > best: best, st, bad = ic, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3: break
    m.load_state_dict(st); m.eval()
    with torch.no_grad(): return m(Xe).numpy(), key


def fit_cls(f):
    _, Tt, yt, _ = samples(f[0], f[1], 1); _, Tv, yv, _ = samples(f[2], f[3], 1); _, Te, _, key = samples(f[4], f[5], 1)
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=200, subsample=0.7,
                           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=2)
    m.fit(Tt.numpy(), (yt.numpy() > 0).astype(int), eval_set=[(Tv.numpy(), (yv.numpy() > 0).astype(int))], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba(Te.numpy())[:, 1] * 2 - 1, key


class Policy(nn.Module):
    def __init__(s):
        super().__init__(); s.rnn = nn.LSTM(5, 24, batch_first=True); s.out = nn.Linear(24, 1)
    def forward(s, x): return torch.tanh(s.out(s.rnn(x)[0])).squeeze(-1)          # position at every step


def fit_pg(f, L=63, cost=0.02):
    """Direct policy search: maximise Sharpe of pos*next_return - cost*|d pos| (cost in vol units)."""
    def wins(a0, a1):
        X, Yw = [], []
        for a in ASSETS:
            for j in range(max(ix(a0), 300) + L, ix(a1), 21):
                X.append(CH[a][j - L + 1:j + 1, :5]); Yw.append(Y[a][j - L + 1:j + 1])
        return torch.tensor(np.array(X)), torch.tensor(np.array(Yw))
    Xt, Yt = wins(f[0], f[1]); Xv, Yv = wins(f[2], f[3])
    def shp(m, X, Yw):
        p = m(X); r = p * Yw - cost * torch.diff(p, dim=1, prepend=p[:, :1] * 0).abs(); r = r[:, 20:]
        return r.mean() / (r.std() + 1e-6)
    m = Policy(); opt = torch.optim.Adam(m.parameters(), 2e-3); best, st, bad = -9, None, 0
    for ep in range(40):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; opt.zero_grad(); (-shp(m, Xt[b], Yt[b])).backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = shp(m, Xv, Yv).item()
        if v > best: best, st, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5: break
    m.load_state_dict(st); m.eval(); out, key = [], []
    with torch.no_grad():
        for a in ASSETS:
            j0, j1 = ix(f[4]), min(ix(f[5]), len(dates) - 2)
            p = m(torch.tensor(CH[a][None, j0 - 250:j1 + 1, :5]))[0, 250:].numpy()
            out += list(p); key += [(a, j) for j in range(j0, j1 + 1)]
    return np.array(out), key


def fit_ddqn(f, gamma=0.9, cost=0.02, steps=6000):
    """Double DQN on logged market data: state = features + current position, actions short/flat/long."""
    def tab(a0, a1):
        S, S2, R = [], [], []
        for a in ASSETS:
            j0, j1 = max(ix(a0), 300), min(ix(a1), len(dates) - 3)
            S.append(TAB[a][j0:j1]); S2.append(TAB[a][j0 + 1:j1 + 1]); R.append(Y[a][j0:j1])
        return torch.tensor(np.concatenate(S)), torch.tensor(np.concatenate(S2)), torch.tensor(np.concatenate(R))
    St, S2t, Rt = tab(f[0], f[1]); Sv, S2v, Rv = tab(f[2], f[3])
    mk = lambda: nn.Sequential(nn.Linear(12, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 3))
    q, qt = mk(), mk(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), 1e-3)
    acts = torch.tensor([-1.0, 0.0, 1.0]); best, st = -9, None
    def greedy_sharpe(S, R):
        with torch.no_grad():
            pos = torch.zeros(1); out = []
            # one pass with the position fed back; vectorised approximation: use previous greedy action
            a_prev = torch.zeros(len(S))
            for _ in range(2):
                a = acts[q(torch.cat([S, a_prev[:, None]], 1)).argmax(1)]; a_prev = torch.roll(a, 1); a_prev[0] = 0
            r = a * R - cost * (a - a_prev).abs(); return (r.mean() / (r.std() + 1e-9)).item(), a
    for it in range(steps):
        b = torch.randint(0, len(St), (512,)); pp = acts[torch.randint(0, 3, (512,))]; ai = torch.randint(0, 3, (512,)); a = acts[ai]
        r = a * Rt[b] - cost * (a - pp).abs()
        with torch.no_grad():
            nxt = torch.cat([S2t[b], a[:, None]], 1); target = r + gamma * qt(nxt).gather(1, q(nxt).argmax(1, keepdim=True)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q(torch.cat([St[b], pp[:, None]], 1)).gather(1, ai[:, None]).squeeze(1), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0: qt.load_state_dict(q.state_dict())
        if it % 500 == 499:
            v, _ = greedy_sharpe(Sv, Rv)
            if v > best: best, st = v, {k: x.clone() for k, x in q.state_dict().items()}
    q.load_state_dict(st); out, key = [], []
    with torch.no_grad():
        for a in ASSETS:
            j0, j1 = ix(f[4]), min(ix(f[5]), len(dates) - 2); pos = 0.0
            Q = lambda j, p: q(torch.tensor(np.append(TAB[a][j], p).astype(np.float32))[None])[0]
            for j in range(j0, j1 + 1):
                pos = float(acts[Q(j, pos).argmax()]); out.append(pos); key.append((a, j))
    return np.array(out), key


MODELS = {"BiLSTM + attention": lambda f: fit_sup(BiLSTMAtt, 5, f), "LSTM-ARIMA hybrid": lambda f: fit_sup(LSTMNet, 6, f),
          "Direction classifier (TATS-style)": fit_cls, "Policy-gradient LSTM (net Sharpe)": fit_pg, "Double DQN": fit_ddqn}
rows, rets = [], {}
for name, fn in MODELS.items():
    t0 = time.time(); F = pd.DataFrame(np.nan, index=dates, columns=ASSETS)
    for blk, f in FITS.items():
        p, key = fn(f)
        for (a, j), v in zip(key, p): F.iat[j, ASSETS.index(a)] = v
    direct = name.startswith(("Policy", "Double"))
    sig = F if direct else F.div(F.stack().groupby(level=0).std().expanding(60).median(), axis=0).clip(-2, 2)
    w = unit_vol((sig / vol).fillna(0.0) / cls_n, cov); net, _, _ = pnl(w, D); g, _, _ = pnl(w, D, 0.0, False)
    rets[name] = net; r = {"model": name, "secs": round(time.time() - t0)}
    for blk, f in FITS.items():
        r[blk + "_net"] = sharpe(net[f[4]:f[5]]); r[blk + "_gross"] = sharpe(g[f[4]:f[5]])
    r["turnover"] = float(w[FITS["VALID"][4]:].diff().abs().sum(axis=1).mean()); rows.append(r); print(r, flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(RES, "q12_dl_rl.csv"), index=False)
pd.DataFrame(rets).to_parquet(os.path.join(RES, "q12_returns.parquet")); print(T.round(2).to_string(index=False))
