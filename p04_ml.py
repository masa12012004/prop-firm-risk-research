"""Does machine learning / deep learning add a forecast the rule-based sleeves lack?

Task     : predict tomorrow's volatility-scaled return for each of the 14 instruments.
Models   : Ridge, LightGBM, MLP, CNN, LSTM, GRU, BiGRU, CNN-LSTM hybrid, Transformer.
Protocol : fit A = train <= 2015, early-stop on 2016-2018  -> predicts VALID (2019 .. 2021-11)
           fit B = train <= 2019, early-stop on 2020-2021-11 -> predicts BROKER (2021-11 .. 2026-10)
           No model ever sees its own evaluation block. Purge of 5 days at each boundary.
Scoring  : daily rank IC, and net Sharpe of a unit-vol book trading the forecast
           (same costs and financing as every other sleeve).
"""
import os, time, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from pf_lib import *

warnings.filterwarnings("ignore")
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
SEQ = 30

D = load_daily()[ASSETS]
cov = ewcov(D)
vol = ewvol(D)
sr = (D / vol.shift(1)).clip(-5, 5)                    # scaled return, known at close t
y = sr.shift(-1)                                        # target: tomorrow's scaled return
usd = sr[FX].mul([-1 if p not in INV else 1 for p in FX], axis=1).mean(axis=1)   # USD strength
mkt = sr[["SP500", "NAS100", "DOW30"]].mean(axis=1)
logp = np.log1p(D.fillna(0)).cumsum()
trend = baz_trend(logp)
vr = (D.rolling(5).std() / D.rolling(60).std())

feat = {}
for a in ASSETS:
    f = pd.DataFrame(index=D.index)
    for l in range(5):
        f[f"r{l}"] = sr[a].shift(l)
    for k in (5, 10, 21, 63, 126, 252):
        f[f"s{k}"] = sr[a].rolling(k).sum() / np.sqrt(k)
    f["trend"] = trend[a]; f["vr"] = vr[a]
    f["mkt1"] = mkt; f["mkt5"] = mkt.rolling(5).sum() / np.sqrt(5)
    f["usd1"] = usd; f["usd5"] = usd.rolling(5).sum() / np.sqrt(5)
    f["gold1"] = sr["XAUUSD"]; f["dow"] = D.index.dayofweek
    for c in ("FX", "IDX", "GOLD"):
        f[f"is_{c}"] = float(CLASS[a] == c)
    f["y"] = y[a]; f["asset"] = a
    feat[a] = f
P = pd.concat(feat.values()).reset_index(names="date").dropna()
P = P[P.date >= "2005-01-01"]
FCOLS = [c for c in P.columns if c not in ("date", "y", "asset")]

# sequence tensor: per asset, channels = own scaled ret, market, usd, gold, vol ratio
chan = {a: np.stack([sr[a], mkt, usd, sr["XAUUSD"], vr[a].clip(0, 5)], axis=1) for a in ASSETS}
pos = {d: i for i, d in enumerate(D.index)}


def seqs(df):
    out = np.zeros((len(df), SEQ, 5), dtype=np.float32)
    for i, (a, d) in enumerate(zip(df.asset.values, df.date.values)):
        j = pos[pd.Timestamp(d)]
        out[i] = np.nan_to_num(chan[a][j - SEQ + 1:j + 1])
    return out


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind, h = kind, 32
        if kind == "MLP":
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(SEQ * 5, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, h), nn.ReLU())
        elif kind == "CNN":
            self.body = nn.Sequential(nn.Conv1d(5, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif kind in ("LSTM", "GRU", "BiGRU"):
            rnn = nn.LSTM if kind == "LSTM" else nn.GRU
            self.rnn = rnn(5, h if kind != "BiGRU" else h // 2, batch_first=True, bidirectional=(kind == "BiGRU"))
        elif kind == "CNN_LSTM":
            self.conv = nn.Sequential(nn.Conv1d(5, 16, 3, padding=1), nn.ReLU())
            self.rnn = nn.LSTM(16, h, batch_first=True)
        elif kind == "Transformer":
            self.inp = nn.Linear(5, h); self.pe = nn.Parameter(torch.zeros(1, SEQ, h))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 64, dropout=0.2, batch_first=True), 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h + 3, 1))

    def forward(self, x, s):
        k = self.kind
        if k == "MLP": z = self.body(x)
        elif k == "CNN": z = self.body(x.transpose(1, 2))
        elif k in ("LSTM", "GRU", "BiGRU"): z = self.rnn(x)[0][:, -1]
        elif k == "CNN_LSTM": z = self.rnn(self.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1]
        else: z = self.enc(self.inp(x) + self.pe)[:, -1]
        return self.head(torch.cat([z, s], 1)).squeeze(1)


def fit_dl(kind, tr, va, te):
    Xt, Xv, Xe = (torch.tensor(seqs(d)) for d in (tr, va, te))
    st = lambda d: torch.tensor(d[["is_FX", "is_IDX", "is_GOLD"]].values, dtype=torch.float32)
    yt, yv = torch.tensor(tr.y.values, dtype=torch.float32), va.y.values
    m = Net(kind); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    best, state, bad = -9, None, 0
    for ep in range(12):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]; opt.zero_grad()
            nn.functional.mse_loss(m(Xt[b], st(tr)[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): ic = spearmanr(m(Xv, st(va)).numpy(), yv)[0]
        if ic > best: best, state, bad = ic, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): return m(Xe, st(te)).numpy(), best


def fit_tab(kind, tr, va, te):
    if kind == "Ridge":
        m = Ridge(alpha=100.0).fit(tr[FCOLS], tr.y); return m.predict(te[FCOLS]), spearmanr(m.predict(va[FCOLS]), va.y)[0]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                          subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=2)
    m.fit(tr[FCOLS], tr.y, eval_set=[(va[FCOLS], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(te[FCOLS]), spearmanr(m.predict(va[FCOLS]), va.y)[0]


FITS = {"VALID": ("2015-12-24", "2016-01-01", "2018-12-24", "2019-01-01", str(VALID_END.date())),
        "BROKER": ("2019-12-24", "2020-01-01", "2021-10-29", str(BROKER_START.date()), "2026-10-30")}
MODELS = ["Ridge", "LightGBM", "MLP", "CNN", "LSTM", "GRU", "BiGRU", "CNN_LSTM", "Transformer"]
preds, rows = {}, []
for name in MODELS:
    t0 = time.time(); pr = []
    for blk, (tr_end, va_a, va_b, te_a, te_b) in FITS.items():
        tr, va, te = P[P.date <= tr_end], P[(P.date >= va_a) & (P.date <= va_b)], P[(P.date >= te_a) & (P.date <= te_b)]
        p, vic = (fit_tab if name in ("Ridge", "LightGBM") else fit_dl)(name, tr, va, te)
        q = te[["date", "asset", "y"]].copy(); q["pred"] = p; q["blk"] = blk; q["val_ic"] = vic; pr.append(q)
    q = pd.concat(pr); preds[name] = q
    F = q.pivot(index="date", columns="asset", values="pred").reindex(D.index)[ASSETS]
    # forecast -> risk-scaled weights. Standardise with an expanding std (no look-ahead).
    sc = F.stack().groupby(level=0).std().expanding(60).median()
    v = (F.div(sc, axis=0).clip(-2, 2) / vol).fillna(0.0)
    cls_n = pd.Series({a: (3 * sum(CLASS[b] == CLASS[a] for b in ASSETS)) for a in ASSETS})
    w = unit_vol(v / cls_n, cov); net, cost, fin = pnl(w, D); g, _, _ = pnl(w, D, 0.0, False)
    preds[name + "_ret"] = net
    row = {"model": name, "secs": round(time.time() - t0)}
    for blk in FITS:
        qq = q[q.blk == blk]; a, b = FITS[blk][3], FITS[blk][4]
        ic = qq.groupby("date").apply(lambda x: spearmanr(x.pred, x.y)[0]).dropna()
        row.update({f"{blk}_val_ic": qq.val_ic.iloc[0], f"{blk}_ic": spearmanr(qq.pred, qq.y)[0],
                    f"{blk}_ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                    f"{blk}_gross_sh": sharpe(g[a:b]), f"{blk}_net_sh": sharpe(net[a:b]),
                    f"{blk}_turnover": float(w[a:b].diff().abs().sum(axis=1).mean())})
    rows.append(row); print(row, flush=True)

# equal-weight forecast ensemble of all models
E = sum(preds[m].set_index(["date", "asset"]).pred.groupby(level=0).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        for m in MODELS)
F = E.unstack().reindex(D.index)[ASSETS]
v = (F.div(F.stack().groupby(level=0).std().expanding(60).median(), axis=0).clip(-2, 2) / vol).fillna(0.0)
cls_n = pd.Series({a: (3 * sum(CLASS[b] == CLASS[a] for b in ASSETS)) for a in ASSETS})
w = unit_vol(v / cls_n, cov); net, _, _ = pnl(w, D); g, _, _ = pnl(w, D, 0.0, False)
row = {"model": "ENSEMBLE_ALL"}
for blk in FITS:
    a, b = FITS[blk][3], FITS[blk][4]
    row.update({f"{blk}_gross_sh": sharpe(g[a:b]), f"{blk}_net_sh": sharpe(net[a:b]),
                f"{blk}_turnover": float(w[a:b].diff().abs().sum(axis=1).mean())})
rows.append(row); preds["ENSEMBLE_ALL_ret"] = net
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "p04_ml.csv"), index=False)
pd.DataFrame({k: v for k, v in preds.items() if k.endswith("_ret")}).to_parquet(os.path.join(RES, "p04_ml_returns.parquet"))
F.to_parquet(os.path.join(RES, "p04_ensemble_forecast.parquet"))
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
