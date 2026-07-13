"""Kernel-weighted trend/covariance estimation and tangency portfolios.

Method (agreed 2026-07-13):
  y_i(t) = log total-return wealth, t in years, monthly observations.
  For lookback H in {5, 10, 20}: exponential kernel q_k = exp(-(T-t_k)/tau),
  tau = H/ln(20) (age-H data gets 5% weight), truncated at age H.
  b_H: WLS slope of y on t per asset -> annual continuously-compounded return.
  C_H: kernel-weighted covariance of overlapping one-year detrended log
       returns eta_k = y(t_k) - y(t_k - 1yr) - b_H (no re-demeaning).
  Lognormal conversion: m_i = exp(b_i + C_ii/2) - 1,
       S_ij = exp(b_i + b_j + (C_ii + C_jj)/2) * (exp(C_ij) - 1).
  Tangency portfolio: max (w'm - rf) / sqrt(w'Sw) over the 4 risky funds,
       long-only, sum(w)=1, rf = exp(b_CASH) - 1 (two-fund separation:
       CASH/FDRXX then sets position along the CML).
  Uncertainty: circular block bootstrap (24-month blocks) of the joint
       monthly log-return vectors within each window; re-run the full
       estimator per replicate.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(20260713)

ASSETS = ["CASH", "FLCOX", "FSKAX", "FXNAX", "FTIHX"]
RISKY = ["FLCOX", "FSKAX", "FXNAX", "FTIHX"]
LOOKBACKS = [5, 10, 20]
N_BOOT = 500
BLOCK = 24  # months

r = pd.read_csv("data/monthly_log_returns.csv", parse_dates=[0], index_col=0)[ASSETS].dropna()
T_end = len(r)


def estimate(rw, H):
    """rw: (n_months, 5) monthly log returns, most recent last. Returns b, C."""
    n = len(rw)
    y = np.vstack([np.zeros(rw.shape[1]), np.cumsum(rw, axis=0)])  # log wealth levels, n+1 points
    t = np.arange(n + 1) / 12.0                                    # years
    age = t[-1] - t                                                # age of each observation
    tau = H / np.log(20.0)
    q = np.exp(-age / tau)
    q[age > H] = 0.0
    # WLS slope per asset
    X = np.column_stack([np.ones_like(t), t])
    W = q
    XtWX = X.T @ (X * W[:, None])
    XtWy = X.T @ (y * W[:, None])
    B = np.linalg.solve(XtWX, XtWy)
    b = B[1]
    # overlapping 1-year detrended log returns, weight at window end
    eta = y[12:] - y[:-12] - b            # (n+1-12, 5)
    qe = q[12:]
    C = (eta * qe[:, None]).T @ eta / qe.sum()
    return b, C


def to_ordinary(b, C):
    m = np.exp(b + 0.5 * np.diag(C)) - 1
    d = np.diag(C)
    S = np.exp(np.add.outer(b, b) + 0.5 * np.add.outer(d, d)) * (np.exp(C) - 1)
    return m, S


def tangency(m, S, rf):
    k = len(m)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    obj = lambda w: -(w @ m - rf) / np.sqrt(w @ S @ w)
    best = None
    for w0 in [np.full(k, 1 / k), np.eye(k)[np.argmax(m)]]:
        res = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def block_bootstrap(rw):
    n = len(rw)
    idx = np.concatenate([(np.arange(BLOCK) + s) % n
                          for s in rng.integers(0, n, int(np.ceil(n / BLOCK)))])[:n]
    return rw[idx]


results = {}
for H in LOOKBACKS:
    window = r.iloc[-min(len(r), H * 12 + 12):]  # H years + 1 for the first eta
    rw = window.values
    b, C = estimate(rw, H)
    m, S = to_ordinary(b, C)
    rf = np.exp(b[0]) - 1
    ridx = [ASSETS.index(a) for a in RISKY]
    w = tangency(m[ridx], S[np.ix_(ridx, ridx)], rf)
    er = w @ m[ridx]
    sd = np.sqrt(w @ S[np.ix_(ridx, ridx)] @ w)

    # bootstrap
    bs_b, bs_w = [], []
    for _ in range(N_BOOT):
        bb, CC = estimate(block_bootstrap(rw), H)
        bs_b.append(bb)
        mm, SS = to_ordinary(bb, CC)
        bs_w.append(tangency(mm[ridx], SS[np.ix_(ridx, ridx)], np.exp(bb[0]) - 1))
    bs_b, bs_w = np.array(bs_b), np.array(bs_w)

    results[H] = dict(b=b, C=C, m=m, S=S, rf=rf, w=w, er=er, sd=sd,
                      b_se=bs_b.std(0), w_q10=np.quantile(bs_w, 0.10, 0),
                      w_med=np.quantile(bs_w, 0.50, 0), w_q90=np.quantile(bs_w, 0.90, 0))

    # ESS diagnostics
    n = len(rw)
    t = np.arange(n + 1) / 12.0
    age = t[-1] - t
    q = np.exp(-age / (H / np.log(20)))
    q[age > H] = 0
    ess = q.sum() ** 2 / (q ** 2).sum()

    spliced = [a for a in ASSETS if a != "CASH" and
               window.index[0] < pd.read_csv("data/splice_dates.csv", index_col=0,
                                             parse_dates=[1]).loc[a].iloc[0]]
    print(f"\n{'=' * 78}\nLOOKBACK H = {H} years   (tau = {H / np.log(20):.2f} yr, ESS = {ess:.0f} months"
          f"{', SPLICED: ' + ','.join(spliced) if spliced else ', all real fund data'})")
    print(f"\nb_{H} — annual log-return trend (WLS slope), with bootstrap SE:")
    for a, bi, se in zip(ASSETS, b, bs_b.std(0)):
        print(f"  {a:6s} {bi:+.4f}  (SE {se:.4f})   -> ordinary {np.exp(bi) - 1:+.2%}")
    print(f"\nC_{H} — one-year log-return covariance (x1e4, i.e. %^2 units):")
    print(pd.DataFrame(C * 1e4, index=ASSETS, columns=ASSETS).round(1).to_string())
    print(f"\nCorrelations:")
    dsd = np.sqrt(np.diag(C))
    print(pd.DataFrame(C / np.outer(dsd, dsd), index=ASSETS, columns=ASSETS).round(2).to_string())
    print(f"\nOrdinary one-year moments: m = {np.array2string(m, formatter={'float': lambda x: f'{x:+.2%}'})}, rf = {rf:.2%}")
    print(f"Tangency portfolio (risky funds, long-only):")
    for a, wi, lo, hi in zip(RISKY, w, results[H]['w_q10'], results[H]['w_q90']):
        print(f"  {a:6s} {wi:6.1%}   bootstrap 10-90%: [{lo:5.1%}, {hi:5.1%}]")
    print(f"  E[R] = {er:.2%}, SD = {sd:.2%}, Sharpe = {(er - rf) / sd:.2f}")

# persist
for H, R in results.items():
    pd.DataFrame({"b": R["b"], "b_se": R["b_se"], "m": R["m"]}, index=ASSETS).to_csv(f"results/moments_b_H{H}.csv")
    pd.DataFrame(R["C"], index=ASSETS, columns=ASSETS).to_csv(f"results/cov_C_H{H}.csv")
    pd.DataFrame(R["S"], index=ASSETS, columns=ASSETS).to_csv(f"results/cov_S_ordinary_H{H}.csv")
    pd.DataFrame({"w": R["w"], "w_q10": R["w_q10"], "w_med": R["w_med"], "w_q90": R["w_q90"]},
                 index=RISKY).to_csv(f"results/tangency_H{H}.csv")
print("\nSaved results/*.csv")
