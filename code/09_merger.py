"""09_merger.py: counterfactual merger simulation (default: Spirit NK + Frontier F9).

Uses the random-coefficients results from 08_estimate_pyblp.py:
  1. recover marginal costs from the Bertrand-Nash first-order conditions,
  2. give the target's products to the acquirer (change who sets prices jointly),
  3. re-solve prices in each market where both firms sell (costs held constant unless --efficiency),
  4. compare prices, quantities, and consumer surplus.
Assumptions to state in the memo: product set and characteristics fixed, no entry or exit,
no cost savings (unless --efficiency), the demand estimates are correct.
"""
import argparse
import numpy as np
import pandas as pd
import pyblp

ap = argparse.ArgumentParser()
ap.add_argument("--acquirer", default="NK")
ap.add_argument("--target", default="F9")
ap.add_argument("--efficiency", type=float, default=0.0, help="cut in marginal cost of the merged firm's products, e.g. 0.05")
args = ap.parse_args()
pyblp.options.verbose = False

res = pyblp.read_pickle("output/results_rc.pickle")
pr = res.problem.products
market = np.asarray(pr.market_ids).flatten().astype(str)
firm = np.asarray(pr.firm_ids).flatten().astype(str)
price = np.asarray(pr.prices).flatten()
share = np.asarray(pr.shares).flatten()
nonstop = np.asarray(pr.nonstop).flatten() if "nonstop" in pr.dtype.names else np.full(len(price), np.nan)
costs = res.compute_costs()                      # (N, 1)
merged = np.where(firm == args.target, args.acquirer, firm)
msize = pd.read_csv("data/clean/pyblp_sample.csv", usecols=["market_ids", "market_size"]).drop_duplicates("market_ids") \
          .set_index("market_ids")["market_size"]

df = pd.DataFrame({"market": market, "firm": firm})
has = df.groupby("market")["firm"].agg(lambda s: {args.acquirer, args.target} <= set(s))
overlap = has[has].index.tolist()
both_nonstop = [m for m in overlap if all(((market == m) & (firm == f) & (nonstop == 1)).any() for f in (args.acquirer, args.target))]
print(f"Markets in the estimation sample: {df['market'].nunique():,}")
print(f"Markets where {args.acquirer} and {args.target} both sell: {len(overlap):,} "
      f"({len(overlap) / df['market'].nunique():.1%}); both with a nonstop product: {len(both_nonstop):,}")
if not overlap:
    raise SystemExit("No overlap markets in this sample. Use more routes (--n-routes) or pick another pair.")

rows, fails = [], 0
for t in overlap:
    idx = np.where(market == t)[0]
    c_t = costs[idx].copy()
    if args.efficiency:
        c_t[np.isin(firm[idx], [args.acquirer, args.target])] *= (1 - args.efficiency)
    try:
        p_new = res.compute_prices(firm_ids=merged[idx].reshape(-1, 1), costs=c_t, market_id=t)
        p_new = np.asarray(p_new).flatten()
        s_new = np.asarray(res.compute_shares(market_id=t, prices=p_new.reshape(-1, 1))).flatten()
        cs_old = res.compute_consumer_surpluses(market_id=t)
        cs_new = res.compute_consumer_surpluses(prices=p_new.reshape(-1, 1), market_id=t)
    except Exception as e:                       # a market where the price equilibrium did not solve
        fails += 1
        continue
    ms = float(msize[t])
    for j, i in enumerate(idx):
        rows.append({"market": t, "firm": firm[i], "nonstop": nonstop[i], "price_old": price[i] * 100,
                     "price_new": float(p_new[j]) * 100, "qty_old": share[i] * ms, "qty_new": float(s_new[j]) * ms,
                     "dCS_market_usd": float(np.squeeze(cs_new) - np.squeeze(cs_old)) * ms * 100})
out = pd.DataFrame(rows)
if out.empty:
    raise SystemExit("Every market failed to solve. Tell me the error from a single market and we will debug.")
out["pct_price_change"] = out["price_new"] / out["price_old"] - 1
out["group"] = np.where(out["firm"].isin([args.acquirer, args.target]), "merging firms",
                        np.where(out["firm"] == "FRINGE", "fringe", "other rivals"))
print(f"Solved {out['market'].nunique():,} overlap markets; {fails} failed to converge")

def wavg(g):
    return np.average(g["pct_price_change"], weights=g["qty_old"])
summ = []
for label, g in [("all overlap markets", out), ("overlap markets with both nonstop", out[out["market"].isin(both_nonstop)])]:
    for grp, h in g.groupby("group"):
        summ.append({"sample": label, "group": grp, "products": len(h),
                     "mean_pct_price_change": h["pct_price_change"].mean(),
                     "qty_weighted_pct_price_change": wavg(h),
                     "p10": h["pct_price_change"].quantile(.1), "p90": h["pct_price_change"].quantile(.9)})
summ = pd.DataFrame(summ)
cs = out.drop_duplicates("market")["dCS_market_usd"]
print("\nPrice effects of the merger (fractions: 0.07 = 7%):\n", (summ.set_index(["sample", "group"]) * 1).round(4).to_string())
print(f"\nConsumer surplus change in overlap markets (approx. dollars per quarter): total {cs.sum():,.0f}, "
      f"mean per market {cs.mean():,.0f}")
tot_old = (out["price_old"] * out["qty_old"]).sum(); tot_new = (out["price_new"] * out["qty_new"]).sum()
print(f"Passenger spending in overlap markets: {tot_new / tot_old - 1:+.2%}; quantity: "
      f"{out['qty_new'].sum() / out['qty_old'].sum() - 1:+.2%}")
tag = f"_eff{int(args.efficiency * 100)}" if args.efficiency else ""
summ.to_csv(f"output/merger_summary{tag}.csv", index=False)
out.to_csv(f"output/merger_by_product{tag}.csv", index=False)

# sanity check: markets without overlap should not change (merged ownership = old ownership there)
non = [m for m in np.unique(market) if m not in set(overlap)][:3]
for t in non:
    idx = np.where(market == t)[0]
    p_chk = res.compute_prices(firm_ids=merged[idx].reshape(-1, 1), costs=costs[idx], market_id=t)
    print(f"Sanity: non-overlap market {t}: max |price change| = {np.abs(p_chk.flatten() - price[idx]).max() * 100:.4f} dollars")
