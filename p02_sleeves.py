"""Evaluate the candidate sleeves, each with parameters fixed from the literature.

Protocol
  DESIGN  2005-01 .. 2018-12  : the only block allowed to influence which sleeves are kept
  VALID   2019-01 .. 2021-11  : frozen check
  BROKER  2021-11 .. 2026-10  : broker H1 data; this is the block the challenge is simulated on
Costs: 1 bp per side + financing (6%/yr on long index/gold notional, 1%/yr on FX gross).
"""
import os
import numpy as np
import pandas as pd
from pf_lib import *

D = load_daily()
cov = ewcov(D[ASSETS])
raw = raw_sleeves(D)
unit = {k: unit_vol(v, cov) for k, v in raw.items()}

# class-level trend pieces, to see where the trend return comes from
for k, cols in (("T_fx", FX), ("T_idx", IDX), ("T_gold", GOLD)):
    v = raw["T"].copy()
    v[[c for c in ASSETS if c not in cols]] = 0.0
    unit[k] = unit_vol(v, cov)

periods = {"DESIGN": ("2005-01-01", DESIGN_END), "VALID": (DESIGN_END + pd.Timedelta(days=1), VALID_END),
           "BROKER": (BROKER_START, "2026-10-30"), "ALL": ("2005-01-01", "2026-10-30")}
rows, rets = [], {}
for k, w in unit.items():
    net, cost, fin = pnl(w, D)
    gross, _, _ = pnl(w, D, cost_mult=0.0, financing=False)
    rets[k] = net
    for pn, (a, b) in periods.items():
        s = stats(net[a:b])
        rows.append({"sleeve": k, "period": pn, **s, "gross_sharpe": sharpe(gross[a:b]),
                     "cost_ann": cost[a:b].mean() * ANN, "fin_ann": fin[a:b].mean() * ANN,
                     "gross_lev": w[a:b].abs().sum(axis=1).mean()})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(RES, "p02_sleeves.csv"), index=False)
pd.DataFrame(rets).to_parquet(os.path.join(RES, "p02_sleeve_returns.parquet"))
pd.set_option("display.width", 250)
show = R.pivot(index="sleeve", columns="period", values="sharpe")[["DESIGN", "VALID", "BROKER", "ALL"]]
print("NET Sharpe by period\n", show.round(2).to_string())
print("\nALL-period detail\n", R[R.period == "ALL"].drop(columns="period").round(3).to_string(index=False))
print("\nBROKER-period detail\n", R[R.period == "BROKER"].drop(columns="period").round(3).to_string(index=False))
X = pd.DataFrame(rets)["2005":]
print("\ncorrelation (ALL)\n", X[["T", "B", "R", "RL", "GL"]].corr().round(2).to_string())
