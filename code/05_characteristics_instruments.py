"""05_characteristics_instruments.py: non-price characteristics and BLP instruments."""
import pandas as pd
import pyblp

# Southwest, JetBlue, Spirit, Frontier, Allegiant, Sun Country (check against printed carriers)
LCC = {"WN", "B6", "NK", "F9", "G4", "SY"}

d = pd.read_csv("data/clean/pyblp_input_2019.csv",
                dtype={"firm_ids": str, "product_ids": str})

parts = d["market_ids"].str.split("_", expand=True)
d["city_a"], d["city_b"], d["yq"] = parts[0], parts[1], parts[2]

# ---- characteristics ----
d["distance_k"] = d["distance"] / 1000
d["distance_k2"] = d["distance_k"] ** 2
d["lcc"] = d["firm_ids"].isin(LCC).astype(int)

# ---- carrier presence at the route's endpoints (hub proxy) ----
cr = d[["firm_ids", "yq", "city_a", "city_b"]].drop_duplicates()
long = pd.concat([cr.rename(columns={"city_a": "city"})[["firm_ids", "yq", "city"]],
                  cr.rename(columns={"city_b": "city"})[["firm_ids", "yq", "city"]]])
deg = long.groupby(["firm_ids", "yq", "city"]).size().rename("deg").reset_index()
for side in ["a", "b"]:
    d = d.merge(deg.rename(columns={"city": f"city_{side}", "deg": f"deg_{side}"}),
                on=["firm_ids", "yq", f"city_{side}"], how="left")
assert d[["deg_a", "deg_b"]].notna().all().all(), "presence merge failed"
d["presence"] = (d["deg_a"] + d["deg_b"]) / 2 / 100   # routes served, per 100

# ---- BLP instruments (distance omitted: it does not vary within a route) ----
form = pyblp.Formulation("0 + nonstop + lcc + presence")
iv = pyblp.build_blp_instruments(form, d)
for k in range(iv.shape[1]):
    d[f"demand_instruments{k}"] = iv[:, k]
print("Number of instruments built:", iv.shape[1])

# ---- relevance check: correlation of price with each instrument ----
iv_cols = [c for c in d.columns if c.startswith("demand_instruments")]
print("\nCorrelation of price with instruments:")
print(d[iv_cols + ["prices"]].corr()["prices"].drop("prices").round(3))

# ---- carriers, for checking the LCC list ----
print("\nCarriers (products, LCC flag):")
print(d.groupby("firm_ids").agg(products=("prices", "size"), lcc=("lcc", "max"))
        .sort_values("products", ascending=False).head(25))

# ---- save ----
base = ["market_ids", "firm_ids", "product_ids", "shares", "prices", "nonstop",
        "distance_k", "distance_k2", "lcc", "presence"]
out = d[base + iv_cols]
assert out.notna().all().all(), "missing values in final data"
out.to_csv("data/clean/pyblp_input_2019_full.csv", index=False)
out[["prices", "shares", "nonstop", "distance_k", "lcc", "presence"]] \
    .describe().T.to_csv("output/summary_stats_full.csv")
print("\nSaved data/clean/pyblp_input_2019_full.csv:", len(out), "rows,", out.shape[1], "columns")
