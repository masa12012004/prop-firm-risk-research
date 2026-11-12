"""Hybrid: does the ML/DL ensemble forecast improve the rule-based book when used as a tilt or a veto?
Both hybrids were specified before looking at their results:
  TILT : weight x (1 + 0.5 * clip(z, -1, 1))      VETO : weight set to 0 when z < -1
z = ensemble forecast for that instrument, standardised with an expanding window."""
import os, numpy as np, pandas as pd
from pf_lib import *
from p05_book import build_books
D = load_daily(); books, cov = build_books(D); W = books["COMBO"]
F = pd.read_parquet(os.path.join(RES, "p04_ensemble_forecast.parquet")).reindex(D.index)[ASSETS]
z = F.div(F.stack().groupby(level=0).std().expanding(60).median(), axis=0)
var = {"BOOK (rules only)": W, "HYBRID tilt": unit_vol(W * (1 + 0.5 * z.clip(-1, 1)).fillna(1.0), cov),
       "HYBRID veto": unit_vol(W * (~(z < -1)).astype(float), cov)}
rows = []
for k, w in var.items():
    net, _, _ = pnl(w, D)
    rows.append({"variant": k, "VALID_sharpe": sharpe(net["2019-01-01":VALID_END]), "BROKER_sharpe": sharpe(net[BROKER_START:]),
                 "BROKER_2021_24H1": sharpe(net[BROKER_START:"2024-06-30"]), "BROKER_2024H2_26": sharpe(net["2024-07-01":])})
R = pd.DataFrame(rows).round(3); R.to_csv(os.path.join(RES, "p04b_hybrid.csv"), index=False); print(R.to_string(index=False))
