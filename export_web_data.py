"""Bundle results into JSON for the summary web page."""
import numpy as np
import pandas as pd
import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ASSETS = ["CASH", "FLCOX", "FSKAX", "FXNAX", "FTIHX"]

r = pd.read_csv("data/monthly_log_returns.csv", parse_dates=[0], index_col=0)[ASSETS].dropna()
y = r.cumsum()
splice = pd.read_csv("data/splice_dates.csv", index_col=0, parse_dates=[1])

out = {
    "dates": [d.strftime("%Y-%m") for d in y.index],
    "logwealth": {a: [round(float(v), 4) for v in y[a]] for a in ASSETS},
    # index of first real-fund month within the series (CASH is real throughout)
    "spliceIdx": {a: (int((y.index < splice.loc[a].iloc[0]).sum()) if a in splice.index else 0)
                  for a in ASSETS},
    "H": {}
}
for H in [5, 10, 20]:
    mom = pd.read_csv(f"results/moments_b_H{H}.csv", index_col=0)
    C = pd.read_csv(f"results/cov_C_H{H}.csv", index_col=0).values
    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    tan = pd.read_csv(f"results/tangency_H{H}.csv", index_col=0)
    fro = pd.read_csv(f"results/frontier_weights_H{H}.csv")
    out["H"][str(H)] = {
        "frontier": {"sd": [round(float(v), 4) for v in fro["target_sd"]],
                     "er": [round(float(v), 4) for v in fro["er"]],
                     "w": {a: [round(float(v), 4) for v in fro[a]] for a in ASSETS}},
        "b": [round(float(v), 4) for v in mom["b"]],
        "se": [round(float(v), 4) for v in mom["b_se"]],
        "m": [round(float(v), 4) for v in mom["m"]],
        "corr": [[round(float(v), 2) for v in row] for row in corr],
        "vol": [round(float(v), 4) for v in sd],
        "tangency": {f: {"w": round(float(tan.loc[f, "w"]), 3),
                         "q10": round(float(tan.loc[f, "w_q10"]), 3),
                         "q90": round(float(tan.loc[f, "w_q90"]), 3)} for f in tan.index},
    }
json.dump(out, open("results/web_data.json", "w"))
print("bytes:", os.path.getsize("results/web_data.json"), "| spliceIdx:", out["spliceIdx"])
