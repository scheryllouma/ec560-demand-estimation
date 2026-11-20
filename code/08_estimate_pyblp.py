"""08_estimate_pyblp.py: first-pass demand estimation with pyBLP, and elasticity diagnostics.

Runs: (1) OLS logit (for contrast), (2) IV logit, (3) market-size sensitivity of the logit,
      (4) random-coefficients logit (optionally with income heterogeneity), (5) elasticities and costs.
Usage:  python code/08_estimate_pyblp.py --n-routes 1500          (fast first pass on a random sample of routes)
        python code/08_estimate_pyblp.py                           (all routes; slow)
        python code/08_estimate_pyblp.py --with-income             (adds income x price heterogeneity)
"""
import argparse
import numpy as np
import pandas as pd
import pyblp
import pyhdfe
import statsmodels.api as sm
from numpy.polynomial.hermite_e import hermegauss

ap = argparse.ArgumentParser()
ap.add_argument("--n-routes", type=int, default=0, help="random sample of routes (0 = all)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--instruments", default="all", choices=["all", "cost_only", "blp_only"])
ap.add_argument("--nodes", type=int, default=3, help="Gauss-Hermite nodes per random coefficient")
ap.add_argument("--with-income", action="store_true")
ap.add_argument("--income-sd", type=float, default=0.6, help="sd of log income WITHIN a market (assumption)")
ap.add_argument("--skip-rc", action="store_true")
ap.add_argument("--no-route-fe", action="store_true", help="drop route fixed effects (NOT recommended: price coefficient had the wrong sign without them)")
args = ap.parse_args()
pyblp.options.digits = 3
pyblp.options.verbose = False

d = pd.read_csv("data/clean/pyblp_input_cp2.csv", dtype={"firm_ids": str, "product_ids": str, "market_ids": str})
if args.n_routes:
    rs = np.random.default_rng(args.seed)
    routes = d["route"].unique()
    pick = rs.choice(routes, min(args.n_routes, len(routes)), replace=False)
    d = d[d["route"].isin(pick)].copy()
print(f"Estimation sample: {len(d):,} products, {d['market_ids'].nunique():,} markets, {d['route'].nunique():,} routes")

# ---------- choose instruments ----------
names = pd.read_csv("output/instrument_names.csv")
if args.instruments == "cost_only":
    names = names[names["meaning"].str.startswith("cost")]
elif args.instruments == "blp_only":
    names = names[names["meaning"].str.startswith("blp")]
if names.empty:
    raise SystemExit("No instruments of that type exist (is data/outside/fuel_cost.csv missing?)")
sel = d[names["column"].tolist()].copy()
d = d.drop(columns=[c for c in d.columns if c.startswith("demand_instruments")])
for k, c in enumerate(names["column"]):
    d[f"demand_instruments{k}"] = sel[c].values
print("Instruments used:", names["meaning"].tolist())
d["clustering_ids"] = pd.factorize(d["route"])[0]
d.to_csv("data/clean/pyblp_sample.csv", index=False)

ROUTE_FE = not args.no_route_fe
if ROUTE_FE:   # route FE absorb distance and other route-level characteristics
    X1 = pyblp.Formulation("0 + prices + nonstop + presence", absorb="C(firm_ids) + C(quarter) + C(route)")
    LIN = ["prices", "nonstop", "presence"]; FES = ["firm_ids", "quarter", "route"]
else:
    X1 = pyblp.Formulation("0 + prices + nonstop + distance_k + distance_k2 + presence", absorb="C(firm_ids) + C(quarter)")
    LIN = ["prices", "nonstop", "distance_k", "distance_k2", "presence"]; FES = ["firm_ids", "quarter"]
print("Fixed effects:", FES)

# ---------- (1) OLS logit, for contrast ----------
def ols_logit(df):
    s0 = 1 - df.groupby("market_ids")["shares"].transform("sum")
    y = (np.log(df["shares"]) - np.log(s0)).to_numpy()
    cols = LIN
    ids = np.column_stack([pd.factorize(df[c])[0] for c in FES])
    M = pyhdfe.create(ids, drop_singletons=False).residualize(np.column_stack([y, df[cols].to_numpy(float)]))
    r = sm.OLS(M[:, 0], M[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": df["clustering_ids"].values})
    return pd.DataFrame({"coef": r.params, "se": r.bse}, index=cols)
ols = ols_logit(d)
print("\n(1) OLS logit (price in $100; ignores endogeneity):\n", ols.round(4))
ols.to_csv("output/logit_ols.csv")

# ---------- (2) IV logit ----------
logit_problem = pyblp.Problem(X1, d)
logit = logit_problem.solve(se_type="clustered")
print("\n(2) IV logit:\n", logit)
open("output/pyblp_logit.txt", "w").write(str(logit))
alpha = float(logit.beta[0, 0])
own_logit = alpha * d["prices"].to_numpy() * (1 - d["shares"].to_numpy())
print(f"IV logit price coefficient: {alpha:.3f} (OLS: {ols.loc['prices', 'coef']:.3f}); "
      f"mean own-price elasticity {own_logit.mean():.2f}; share with |elasticity| < 1: {(own_logit > -1).mean():.1%}")

# ---------- (3) market-size sensitivity of the logit ----------
mq = d.groupby("market_ids")["quantity"].transform("sum")
rows = []
for mult in [0.5, 1.0, 2.0]:
    ms = np.maximum(mult * d["market_size"], mq / 0.95)
    dd = d.copy(); dd["shares"] = dd["quantity"] / ms
    r = pyblp.Problem(X1, dd).solve(se_type="clustered")
    a = float(r.beta[0, 0]); e = a * dd["prices"] * (1 - dd["shares"])
    rows.append({"market_size_multiplier": mult, "price_coef": a, "mean_own_elasticity": e.mean(),
                 "median_own_elasticity": e.median(), "mean_outside_share": (1 - dd.groupby("market_ids")["shares"].sum()).mean()})
sens = pd.DataFrame(rows)
print("\n(3) Logit sensitivity to market size:\n", sens.round(3).to_string(index=False))
sens.to_csv("output/logit_market_size_sensitivity.csv", index=False)

if args.skip_rc:
    raise SystemExit("--skip-rc given; stopping after the logit.")

# ---------- (4) random coefficients ----------
pyblp.options.verbose = True
X2 = pyblp.Formulation("1 + prices + nonstop")
sigma0 = np.diag([1.0, 0.5, 1.0])
sigma_bounds = (np.zeros((3, 3)), np.diag([6.0, 4.0, 6.0]))
opt = pyblp.Optimization("l-bfgs-b", {"gtol": 1e-4})

if args.with_income:
    base = pyblp.build_integration(pyblp.Integration("product", args.nodes), 3)
    zi, wi = hermegauss(5); wi = wi / wi.sum()
    mk = d.drop_duplicates("market_ids")[["market_ids", "inc_mean"]].copy()
    mk["inc_mean"] = mk["inc_mean"].fillna(mk["inc_mean"].median())
    blocks = []
    nb = base.nodes.shape[0]
    for m, inc in zip(mk["market_ids"], mk["inc_mean"]):
        nodes = np.repeat(base.nodes, 5, axis=0)
        w = np.repeat(base.weights.flatten(), 5) * np.tile(wi, nb)
        income = inc / 10000 * np.exp(args.income_sd * np.tile(zi, nb) - args.income_sd ** 2 / 2)
        blk = pd.DataFrame(nodes, columns=[f"nodes{k}" for k in range(3)])
        blk["market_ids"] = m; blk["weights"] = w; blk["income"] = income
        blocks.append(blk)
    agents = pd.concat(blocks, ignore_index=True)
    problem = pyblp.Problem((X1, X2), d, pyblp.Formulation("0 + income"), agents)
    pi0 = np.array([[0.0], [0.3], [0.0]])            # income shifts the price coefficient only
    rc = problem.solve(sigma0, pi0, sigma_bounds=sigma_bounds, optimization=opt, method="1s", se_type="clustered")
else:
    problem = pyblp.Problem((X1, X2), d, integration=pyblp.Integration("product", args.nodes))
    rc = problem.solve(sigma0, sigma_bounds=sigma_bounds, optimization=opt, method="1s", se_type="clustered")
print("\n(4) Random-coefficients logit:\n", rc)
open("output/pyblp_rc.txt", "w").write(str(rc))

# ---------- (5) elasticities, costs, markups ----------
el = rc.extract_diagonals(rc.compute_elasticities()).flatten()
costs = rc.compute_costs().flatten()
p = d["prices"].to_numpy()
d["own_elasticity"], d["mc"], d["markup"] = el, costs, (p - costs) / p
q = d["own_elasticity"].quantile([0.1, 0.5, 0.9])
print(f"\nOwn-price elasticities: mean {el.mean():.2f}, median {q[0.5]:.2f}, p10 {q[0.1]:.2f}, p90 {q[0.9]:.2f}")
print(f"Share with |elasticity| < 1 (implies negative marginal cost): {(el > -1).mean():.1%}")
print(f"Marginal cost: share negative {(costs < 0).mean():.1%}; mean markup (p-mc)/p {d['markup'].mean():.2f}")
summ = []
for label, mask in [("all", np.ones(len(d), bool)), ("nonstop", d["nonstop"] == 1),
                    ("connecting", d["nonstop"] == 0)]:
    e = d.loc[mask, "own_elasticity"]
    summ.append({"group": label, "n": int(mask.sum()), "mean": e.mean(), "median": e.median(),
                 "p10": e.quantile(.1), "p90": e.quantile(.9), "share_inelastic": (e > -1).mean(),
                 "mean_markup": d.loc[mask, "markup"].mean()})
for c, g in d.groupby("firm_ids"):
    e = g["own_elasticity"]
    summ.append({"group": c, "n": len(g), "mean": e.mean(), "median": e.median(), "p10": e.quantile(.1),
                 "p90": e.quantile(.9), "share_inelastic": (e > -1).mean(), "mean_markup": g["markup"].mean()})
summ = pd.DataFrame(summ)
print(summ.round(3).to_string(index=False))
summ.to_csv("output/elasticities_summary.csv", index=False)
d[["market_ids", "product_ids", "firm_ids", "own_elasticity", "mc", "markup"]].to_csv("output/product_elasticities_costs.csv", index=False)
rc.to_pickle("output/results_rc.pickle")
print("Saved results_rc.pickle and the tables in output/")
