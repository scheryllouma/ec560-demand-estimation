"""03_build_pyblp_input.py: market size, shares, pyBLP-ready data (baseline).

Baseline market size (data-based, no outside data):
  M(route) = MULT x max over quarters of the route's total quantity.
A route is an undirected city-market pair, so M is constant across quarters.
"""
import pandas as pd

MULTS = [1.25, 2.0, 4.0]
BASE_MULT = 2.0

prod = pd.read_csv("data/clean/products_2019.csv",
                   dtype={"TkCarrier": str, "firm_ids": str, "product_ids": str})
prod["route"] = prod["market_ids"].str.rsplit("_", n=1).str[0]

prod["mkt_quantity"] = prod.groupby("market_ids")["quantity"].transform("sum")
prod["route_peak"] = prod.groupby("route")["mkt_quantity"].transform("max")

# ---- sensitivity of the outside good to the multiplier ----
mk = prod.drop_duplicates("market_ids")
rows = []
for m in MULTS:
    inside = mk["mkt_quantity"] / (m * mk["route_peak"])
    rows.append({"multiplier": m,
                 "mean_inside_share": inside.mean(),
                 "median_inside_share": inside.median(),
                 "max_inside_share": inside.max()})
sens = pd.DataFrame(rows)
print("Sensitivity of inside share to the market-size multiplier:\n", sens)
sens.to_csv("output/market_size_sensitivity.csv", index=False)

# ---- baseline market size and shares ----
prod["market_size"] = BASE_MULT * prod["route_peak"]
prod["shares"] = prod["quantity"] / prod["market_size"]

# ---- pyBLP input checks ----
req = ["market_ids", "firm_ids", "product_ids", "shares", "prices"]
assert prod[req].notna().all().all(), "missing values in required columns"
assert (prod["shares"] > 0).all(), "non-positive shares"
inside_sum = prod.groupby("market_ids")["shares"].sum()
assert (inside_sum < 1).all(), "inside shares sum to >= 1 in some market"
dup = prod.duplicated(["market_ids", "product_ids"]).sum()
print("\nDuplicate (market, product) pairs:", dup)
print("Inside share by market:\n", inside_sum.describe())

# ---- save ----
cols = ["market_ids", "firm_ids", "product_ids", "shares", "prices",
        "nonstop", "distance", "quantity", "market_size"]
prod[cols].to_csv("data/clean/pyblp_input_2019.csv", index=False)

summary = prod[["prices", "shares", "quantity", "distance", "nonstop"]].describe().T
summary.to_csv("output/summary_stats.csv")
print("\nSummary statistics:\n", summary)
print("\nSaved data/clean/pyblp_input_2019.csv:", len(prod), "products,",
      prod["market_ids"].nunique(), "markets")
