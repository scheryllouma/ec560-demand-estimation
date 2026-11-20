"""07_instruments_firststage.py: characteristics, instruments, and the quasi first stage.

Reads  data/clean/products_2019_ms.csv   (from 06_market_size.py)
       data/outside/fuel_cost.csv        (OPTIONAL: columns quarter, fuel_price, and optionally carrier)
Writes data/clean/pyblp_input_cp2.csv, output/first_stage.csv, output/instrument_names.csv

Cost instruments (need the fuel file):
  cost_own   = fuel price x distance                      (own input cost; carrier-specific if the file has carrier)
  cost_rival = sum of RIVAL firms' fuel prices x distance (rivals' input cost shifts competition for you)
  cost_conn  = fuel price x distance x connecting         (connecting trips fly extra miles)
BLP instruments (characteristics of other products), built with pyblp.
"""
import argparse
import os
import numpy as np
import pandas as pd
import pyblp
import pyhdfe
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--route-fe", action="store_true", help="also absorb route fixed effects (stricter)")
args = ap.parse_args()

LCC = {"WN", "B6", "NK", "F9", "G4", "SY"}
d = pd.read_csv("data/clean/products_2019_ms.csv", dtype={"TkCarrier": str, "firm_ids": str, "product_ids": str})
d["prices_usd"] = d["prices"]

# ---------- characteristics ----------
d["distance_k"] = d["distance"] / 1000
d["distance_k2"] = d["distance_k"] ** 2
d["lcc"] = d["firm_ids"].isin(LCC).astype(int)
cr = d.loc[d["firm_ids"] != "FRINGE", ["firm_ids", "quarter", "route", "city_a", "city_b"]].drop_duplicates()
long = pd.concat([cr.rename(columns={"city_a": "city"})[["firm_ids", "quarter", "route", "city"]],
                  cr.rename(columns={"city_b": "city"})[["firm_ids", "quarter", "route", "city"]]])
deg = long.groupby(["firm_ids", "quarter", "city"]).route.nunique().rename("deg").reset_index()
for s in ["a", "b"]:
    d = d.merge(deg.rename(columns={"city": f"city_{s}", "deg": f"deg_{s}"}),
                on=["firm_ids", "quarter", f"city_{s}"], how="left")
d["presence"] = ((d["deg_a"].fillna(0) + d["deg_b"].fillna(0)) / 2) / 100   # routes served per 100; fringe = 0
d = d.drop(columns=["deg_a", "deg_b"])

# ---------- cost instruments ----------
cost_cols = []
fuel_path = "data/outside/fuel_cost.csv"
if os.path.exists(fuel_path):
    f = pd.read_csv(fuel_path)
    f["quarter"] = f["quarter"].astype(str)
    f.loc[~f["quarter"].str.startswith("2019"), "quarter"] = "2019Q" + f["quarter"].str[-1]
    if "carrier" in f.columns:
        d = d.merge(f[["carrier", "quarter", "fuel_price"]].rename(columns={"carrier": "firm_ids"}),
                    on=["firm_ids", "quarter"], how="left")
        d["fuel"] = d["fuel_price"].fillna(d.groupby("quarter")["fuel_price"].transform("mean"))
    else:
        d = d.merge(f[["quarter", "fuel_price"]], on="quarter", how="left")
        d["fuel"] = d["fuel_price"]
    assert d["fuel"].notna().all(), "fuel price missing for some quarters; check fuel_cost.csv"
    d["cost_own"] = d["fuel"] * d["distance_k"]
    own_firm_fuel = d.groupby(["market_ids", "firm_ids"])["fuel"].transform("sum")
    mkt_fuel = d.groupby("market_ids")["fuel"].transform("sum")
    d["cost_rival"] = (mkt_fuel - own_firm_fuel) * d["distance_k"]
    d["cost_conn"] = d["fuel"] * d["distance_k"] * (1 - d["nonstop"])
    cost_cols = ["cost_own", "cost_rival", "cost_conn"]
else:
    print("WARNING: data/outside/fuel_cost.csv not found, so NO cost instruments were built.")

# ---------- BLP instruments ----------
form = pyblp.Formulation("0 + nonstop + lcc + presence")
iv = pyblp.build_blp_instruments(form, d.rename(columns={"market_ids": "market_ids"}))
blp_cols = [f"blp{k}" for k in range(iv.shape[1])]
for k, c in enumerate(blp_cols):
    d[c] = iv[:, k]

all_iv = cost_cols + blp_cols
for k, c in enumerate(all_iv):
    d[f"demand_instruments{k}"] = d[c]
names = pd.DataFrame({"column": [f"demand_instruments{k}" for k in range(len(all_iv))], "meaning": all_iv})
names.to_csv("output/instrument_names.csv", index=False)
print(names.to_string(index=False))

d["clustering_ids"] = pd.factorize(d["route"])[0]
d["prices_hundreds"] = d["prices_usd"] / 100

# ---------- quasi first stage ----------
controls = ["nonstop", "distance_k", "distance_k2", "presence"]
fe_cols = ["firm_ids", "quarter"] + (["route"] if args.route_fe else [])
print("\nQuasi first stage: price on instruments | controls", controls, "| FE", fe_cols, "| clustered by route")

def first_stage(z_cols, label):
    cols = ["prices_hundreds"] + controls + z_cols
    ids = np.column_stack([pd.factorize(d[c])[0] for c in fe_cols])
    M = pyhdfe.create(ids, drop_singletons=False).residualize(d[cols].to_numpy(float))
    y, X, Z = M[:, [0]], M[:, 1:1 + len(controls)], M[:, 1 + len(controls):]
    full = sm.OLS(y, np.column_stack([X, Z])).fit(cov_type="cluster", cov_kwds={"groups": d["clustering_ids"].values})
    base = sm.OLS(y, X).fit()
    q = len(z_cols)
    R = np.zeros((q, X.shape[1] + q)); R[:, X.shape[1]:] = np.eye(q)
    w = full.wald_test(R, use_f=True, scalar=True)
    rows = [{"set": label, "instrument": c, "coef": full.params[len(controls) + i],
             "se": full.bse[len(controls) + i], "t": full.tvalues[len(controls) + i]} for i, c in enumerate(z_cols)]
    summary = {"set": label, "n_instruments": q, "partial_F": float(np.squeeze(w.fvalue)),
               "p_value": float(np.squeeze(w.pvalue)),
               "partial_R2": 1 - full.ssr / base.ssr, "n_obs": len(d)}
    return rows, summary

rows, sums = [], []
sets = {"all": all_iv, "blp_only": blp_cols}
if cost_cols:
    sets["cost_only"] = cost_cols
for label, cols in sets.items():
    r, s = first_stage(cols, label)
    rows += r; sums.append(s)
coef = pd.DataFrame(rows)
summ = pd.DataFrame(sums)
print("\nJoint relevance of excluded instruments:\n", summ.round(4).to_string(index=False))
print("\nCoefficients (price in $100 units):\n", coef.round(4).to_string(index=False))
coef.to_csv("output/first_stage_coefficients.csv", index=False)
summ.to_csv("output/first_stage.csv", index=False)

keep = ["market_ids", "route", "quarter", "firm_ids", "product_ids", "shares", "prices_usd", "prices_hundreds",
        "nonstop", "distance_k", "distance_k2", "lcc", "presence", "market_size", "quantity",
        "inc_mean", "clustering_ids"] + [f"demand_instruments{k}" for k in range(len(all_iv))]
out = d[keep].copy()
out["prices"] = out["prices_hundreds"]            # pyBLP price variable, in $100 units
assert out[["market_ids", "firm_ids", "product_ids", "shares", "prices"]].notna().all().all()
assert not out.duplicated(["market_ids", "product_ids"]).any(), "duplicate market/product pairs"
out.to_csv("data/clean/pyblp_input_cp2.csv", index=False)
out[["prices_usd", "shares", "nonstop", "distance_k", "lcc", "presence"]].describe().T.to_csv("output/summary_stats_cp2.csv")
print("\nSaved data/clean/pyblp_input_cp2.csv:", len(out), "rows,", out.shape[1], "columns")
