"""02_clean.py: filter DB1B Market data and build product-level data.

Market  = undirected city-market pair x quarter
Product = ticketing carrier x {nonstop, connecting} within a market
"""
import glob
import numpy as np
import pandas as pd

SAMPLE_SCALE = 10      # DB1B is a 10% sample in 2019
MIN_FARE, MAX_FARE = 25, 2000
MIN_SAMPLED_PAX = 20   # minimum sampled passengers per product-market

WANT = ["Year", "Quarter", "MktCoupons", "OriginCityMarketID", "DestCityMarketID",
        "OriginState", "DestState", "TkCarrier", "BulkFare", "Passengers",
        "MktFare", "MktDistance", "NonStopMiles"]

# ---------- load all quarters ----------
files = sorted(glob.glob("data/raw/db1b_market_2019_q*.zip"))
print("Files found:", files)
df = pd.concat(
    [pd.read_csv(f, usecols=lambda c: c in WANT, dtype={"TkCarrier": str},
                 low_memory=False) for f in files],
    ignore_index=True)
print("Columns loaded:", list(df.columns))

# ---------- filter bookkeeping ----------
log = []
def record(step, d):
    log.append({"step": step, "rows": len(d), "passengers": d["Passengers"].sum()})
    print(f"{step:<45} rows={len(d):>12,}  pax={d['Passengers'].sum():>14,.0f}")

record("0. Raw data", df)

df = df.dropna(subset=["MktFare", "Passengers", "TkCarrier",
                       "OriginCityMarketID", "DestCityMarketID"])
record("1. Drop missing key fields", df)

if "OriginState" in df.columns:
    out = ["AK", "HI", "PR", "VI", "TT", "GU", "MP", "AS"]
    df = df[~df["OriginState"].isin(out) & ~df["DestState"].isin(out)]
    record("2. Continental US only", df)

if "BulkFare" in df.columns:
    df = df[df["BulkFare"] == 0]
    record("3. Drop bulk fares", df)

df = df[(df["MktFare"] >= MIN_FARE) & (df["MktFare"] <= MAX_FARE)]
record(f"4. Fare between ${MIN_FARE} and ${MAX_FARE}", df)

df = df[df["MktCoupons"] <= 3]
record("5. At most 2 connections", df)

df = df[df["TkCarrier"] != "99"]
record("6. Drop interline (TkCarrier = 99)", df)

# ---------- market and product ids ----------
lo = df[["OriginCityMarketID", "DestCityMarketID"]].min(axis=1).astype(int)
hi = df[["OriginCityMarketID", "DestCityMarketID"]].max(axis=1).astype(int)
df = df[lo != hi].copy()           # drop same-city-market records
lo, hi = lo[df.index], hi[df.index]
df["market_ids"] = (lo.astype(str) + "_" + hi.astype(str) + "_"
                    + df["Year"].astype(str) + "Q" + df["Quarter"].astype(str))
df["nonstop"] = (df["MktCoupons"] == 1).astype(int)
df["fare_x_pax"] = df["MktFare"] * df["Passengers"]
df["dist_x_pax"] = df["MktDistance"] * df["Passengers"]

# ---------- aggregate to product level ----------
prod = (df.groupby(["market_ids", "TkCarrier", "nonstop"], as_index=False)
          .agg(fare_x_pax=("fare_x_pax", "sum"),
               dist_x_pax=("dist_x_pax", "sum"),
               sampled_pax=("Passengers", "sum")))
prod["prices"] = prod["fare_x_pax"] / prod["sampled_pax"]
prod["distance"] = prod["dist_x_pax"] / prod["sampled_pax"]
prod["quantity"] = prod["sampled_pax"] * SAMPLE_SCALE
prod = prod.drop(columns=["fare_x_pax", "dist_x_pax"])
record_df = prod.rename(columns={"sampled_pax": "Passengers"})
record("7. Products before small-cell filter", record_df)

prod = prod[prod["sampled_pax"] >= MIN_SAMPLED_PAX]
record("8. Products with >= %d sampled pax" % MIN_SAMPLED_PAX,
       prod.rename(columns={"sampled_pax": "Passengers"}))

prod["firm_ids"] = prod["TkCarrier"]
prod["product_ids"] = prod["TkCarrier"] + "_" + prod["nonstop"].astype(str)

# ---------- save ----------
pd.DataFrame(log).to_csv("output/filter_counts.csv", index=False)
prod.to_csv("data/clean/products_2019.csv", index=False)

print("\nProducts:", f"{len(prod):,}", "| Markets:", f"{prod['market_ids'].nunique():,}")
print("Products per market:\n", prod.groupby("market_ids").size().describe())
print("\nPrice summary:\n", prod["prices"].describe())
print("\nNonstop share of products:", prod["nonstop"].mean().round(3))
