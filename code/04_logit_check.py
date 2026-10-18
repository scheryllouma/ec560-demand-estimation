"""04_logit_check.py: OLS logit sanity check on the pyBLP input (not the final model)."""
import numpy as np
import pandas as pd

d = pd.read_csv("data/clean/pyblp_input_2019.csv",
                dtype={"firm_ids": str, "product_ids": str})
d["s0"] = 1 - d.groupby("market_ids")["shares"].transform("sum")
d["y"] = np.log(d["shares"]) - np.log(d["s0"])

X = np.column_stack([np.ones(len(d)), d["prices"] / 100,
                     d["nonstop"], d["distance"] / 1000])
b, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
for name, coef in zip(["constant", "price ($100)", "nonstop", "distance (1000 mi)"], b):
    print(f"{name:<20} {coef: .3f}")
print("\nMean outside share:", d["s0"].mean().round(3))
