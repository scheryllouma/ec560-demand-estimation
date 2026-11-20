"""08b_sign_diagnostic.py: why is the price coefficient positive? Quick logit 2SLS under different fixed effects.

Logit: ln(s_j) - ln(s_0) = alpha*price + controls + FE + error.  alpha should be NEGATIVE.
Prints alpha (price in $100) for OLS and IV in several specifications. No pyBLP, runs in about a minute.
"""
import numpy as np
import pandas as pd
import pyhdfe

d = pd.read_csv("data/clean/pyblp_input_cp2.csv", dtype={"firm_ids": str, "product_ids": str, "market_ids": str})
d["s0"] = 1 - d.groupby("market_ids")["shares"].transform("sum")
d["y"] = np.log(d["shares"]) - np.log(d["s0"])
names = pd.read_csv("output/instrument_names.csv")
cost = names.loc[names["meaning"].str.startswith("cost"), "column"].tolist()
blp = names.loc[names["meaning"].str.startswith("blp"), "column"].tolist()
d["route_id"] = pd.factorize(d["route"])[0]
d["firm_id"] = pd.factorize(d["firm_ids"])[0]
d["q_id"] = pd.factorize(d["quarter"])[0]
d["mkt_id"] = pd.factorize(d["market_ids"])[0]
d["firmroute_id"] = pd.factorize(d["firm_ids"] + "_" + d["route"])[0]
cl = d["route_id"].values


def tsls(df, fe_cols, exog, inst):
    """Residualize on FE, then OLS (inst=None) or 2SLS. Returns alpha, clustered se, n."""
    cols = ["prices"] + exog + inst + ["y"]
    M = df[cols].values.astype(float)
    if fe_cols:
        ids = df[fe_cols].values
        M = pyhdfe.create(ids, drop_singletons=False).residualize(M)
    p, E, Z, y = M[:, [0]], M[:, 1:1 + len(exog)], M[:, 1 + len(exog):-1], M[:, [-1]]
    X = np.hstack([p, E])
    if inst:
        W = np.hstack([E, Z])
        Xh = W @ np.linalg.lstsq(W, X, rcond=None)[0]
    else:
        Xh = X
    A = np.linalg.inv(Xh.T @ X)
    b = A @ (Xh.T @ y)
    e = y - X @ b
    g = df["route_id"].values
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(g):
        i = g == c
        s = Xh[i].T @ e[i]
        meat += s @ s.T
    V = A @ meat @ A.T
    return float(b[0, 0]), float(np.sqrt(V[0, 0])), len(df)


specs = [
    ("A. firm+quarter FE (what 08 uses)", ["firm_id", "q_id"], ["nonstop", "distance_k", "distance_k2", "presence"], d),
    ("B. + route FE", ["firm_id", "q_id", "route_id"], ["nonstop", "presence"], d),
    ("C. route FE, drop FRINGE", ["firm_id", "q_id", "route_id"], ["nonstop", "presence"], d[d["firm_ids"] != "FRINGE"]),
    ("D. market (route x quarter) FE + firm FE", ["firm_id", "mkt_id"], ["nonstop"], d),
    ("E. market FE + firm-by-route FE", ["firmroute_id", "mkt_id"], ["nonstop"], d),
]
rows = []
for label, fe, ex, df in specs:
    for iname, inst in [("OLS", []), ("IV all", cost + blp), ("IV cost only", cost), ("IV blp only", blp)]:
        try:
            a, se, n = tsls(df, fe, ex, inst)
            rows.append({"spec": label, "estimator": iname, "alpha": round(a, 3), "se": round(se, 3), "n": n})
        except Exception as ex_:
            print("  (spec failed:", label, iname, repr(ex_)[:150], ")")
            rows.append({"spec": label, "estimator": iname, "alpha": np.nan, "se": np.nan, "n": len(df)})
out = pd.DataFrame(rows)
print("\nPrice coefficient alpha ($100 units); should be NEGATIVE:\n")
print(out.to_string(index=False))
out.to_csv("output/sign_diagnostic.csv", index=False)
