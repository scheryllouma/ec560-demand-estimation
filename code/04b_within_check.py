"""04b_within_check.py: logit within markets (market fixed effects), sanity check only."""
import numpy as np
import pandas as pd

d = pd.read_csv("data/clean/pyblp_input_2019.csv",
                dtype={"firm_ids": str, "product_ids": str})
d["s0"] = 1 - d.groupby("market_ids")["shares"].transform("sum")
d["y"] = np.log(d["shares"]) - np.log(d["s0"])
d["price100"] = d["prices"] / 100

for c in ["y", "price100", "nonstop"]:
    d[c + "_w"] = d[c] - d.groupby("market_ids")[c].transform("mean")

multi = d.groupby("market_ids")["y"].transform("size") > 1
print("Products in markets with 2+ products:", int(multi.sum()), "of", len(d))
X = np.column_stack([d.loc[multi, "price100_w"], d.loc[multi, "nonstop_w"]])
b, *_ = np.linalg.lstsq(X, d.loc[multi, "y_w"].to_numpy(), rcond=None)
print("price ($100):", round(b[0], 3), "| nonstop:", round(b[1], 3))
