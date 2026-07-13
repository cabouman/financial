"""5-asset efficient frontier including CASH as an investable asset.

Answers: once cash is available, does FXNAX still get defensive weight?
For each lookback H, sweep target one-year SD and maximize expected ordinary
return w'm s.t. sqrt(w'Sw) <= target, long-only, fully invested, using the
lognormal-converted moments from estimate.py.
Writes results/frontier_weights_H{5,10,20}.csv.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ASSETS = ["CASH", "FLCOX", "FSKAX", "FXNAX", "FTIHX"]

for H in [5, 10, 20]:
    mom = pd.read_csv(f"results/moments_b_H{H}.csv", index_col=0)
    m = mom["m"].values
    S = pd.read_csv(f"results/cov_S_ordinary_H{H}.csv", index_col=0).values
    n = len(m)

    # minimum-variance SD sets the left edge of the sweep
    cons_base = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    mv = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), method="SLSQP",
                  bounds=[(0, 1)] * n, constraints=cons_base,
                  options={"maxiter": 1000, "ftol": 1e-14})
    sd_min = np.sqrt(mv.fun)
    sd_max = np.sqrt(S[ASSETS.index("FSKAX"), ASSETS.index("FSKAX")])

    rows = []
    w0 = mv.x
    for sd in np.linspace(sd_min * 1.001, sd_max, 25):
        cons = cons_base + [{"type": "ineq", "fun": lambda w, sd=sd: sd**2 - w @ S @ w}]
        res = minimize(lambda w: -(w @ m), w0, method="SLSQP",
                       bounds=[(0, 1)] * n, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-14})
        w = np.clip(res.x, 0, 1); w /= w.sum()
        w0 = w  # warm start along the frontier
        rows.append([sd, w @ m, *w])
    df = pd.DataFrame(rows, columns=["target_sd", "er", *ASSETS])
    df.to_csv(f"results/frontier_weights_H{H}.csv", index=False)

    print(f"\nH = {H}: cash b = {mom.loc['CASH','b']:+.4f} (SE {mom.loc['CASH','b_se']:.4f}), "
          f"FXNAX b = {mom.loc['FXNAX','b']:+.4f} (SE {mom.loc['FXNAX','b_se']:.4f})")
    print("Optimal weights along the frontier (selected target SDs):")
    print(f"  {'SD':>5s} {'E[R]':>6s}  " + "  ".join(f"{a:>6s}" for a in ASSETS))
    for _, r in df.iloc[::5].iterrows():
        print(f"  {r['target_sd']:5.1%} {r['er']:6.1%}  " + "  ".join(f"{r[a]:6.1%}" for a in ASSETS))
    fx_max = df["FXNAX"].max()
    print(f"  max FXNAX weight anywhere on this frontier: {fx_max:.1%}")
