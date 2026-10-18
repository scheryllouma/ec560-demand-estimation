"""02_clean.py: filter DB1B Market data and build product-level data.

Market  = undirected city-market pair x quarter
Product = ticketing carrier x {nonstop, connecting} within a market
"""
import glob
import re
import pandas as pd

SAMPLE_SCALE = 10      # DB1B is a 10% sample in 2019
MIN_FARE, MAX_FARE = 25, 2000
MIN_SAMPLED_PAX = 20   # minimum sampled passengers per product-market

import zipfile

def read_member(path, **kw):
    """Read the CSV inside a BTS zip (some zips also contain readme.html)."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as fh:
            return pd.read_csv(fh, **kw)

def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())

# normalized file column name -> name used in this script
ALIAS = {
    "year": "Year", "quarter": "Quarter",
    "marketcoupons": "MktCoupons", "mktcoupons": "MktCoupons",
    "origincitymarketid": "OriginCityMarketID",
    "destcitymarketid": "DestCityMarketID",
    "originstate": "OriginState", "originstateabr": "OriginState",
    "deststate": "DestState", "deststateabr": "DestState",
    "ticketcarrier": "TkCarrier", "tkcarrier": "TkCarrier",
    "bulkfare": "BulkFare", "passengers": "Passengers",
    "marketfare": "MktFare", "mktfare": "MktFare",
    "marketdistance": "MktDistance", "mktdistance": "MktDistance",
    "nonstopmiles": "NonStopMiles",
}
REQUIRED = ["Year", "Quarter", "MktCoupons", "OriginCityMarketID",
            "DestCityMarketID", "TkCarrier", "BulkFare", "Passengers",
            "MktFare", "MktDistance"]

# ---------- load all quarters ----------
files = sorted(glob.glob("data/raw/db1b_market_2019_q*.zip"))
print("Files found:", files)
if not files:
    raise SystemExit("No files in data/raw/ matching db1b_market_2019_q*.zip")

parts = []
for f in files:
    header = read_member(f, nrows=0).columns
    print("File columns:", list(header))
    d = read_member(f, usecols=lambda c: norm(c) in ALIAS, low_memory=False)
    d = d.rename(columns={c: ALIAS[norm(c)] for c in d.columns})
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
print("Columns used:", list(df.columns))

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}. "
                     f"Check the 'File columns' line above and tell me the names.")

df["TkCarrier"] = df["TkCarrier"].astype(str).str.strip()

# ---------- filter bookkeeping ----------
log = []
def record(step, d):
    pax = d["Passengers"].sum()
    log.append({"step": step, "rows": len(d), "passengers": pax})
    print(f"{step:<45} rows={len(d):>12,}  pax={pax:>14,.0f}")

record("0. Raw data", df)

df = df.dropna(subset=["MktFare", "Passengers", "TkCarrier",
                       "OriginCityMarketID", "DestCityMarketID"])
record("1. Drop missing key fields", df)

if "OriginState" in df.columns and "DestState" in df.columns:
    out = ["AK", "HI", "PR", "VI", "TT", "GU", "MP", "AS"]
    df = df[~df["OriginState"].isin(out) & ~df["DestState"].isin(out)]
    record("2. Continental US only", df)
else:
    print("2. Continental US filter SKIPPED: no state columns in this file")

df = df[df["BulkFare"] == 0]
record("3. Drop bulk fares", df)

df = df[(df["MktFare"] >= MIN_FARE) & (df["MktFare"] <= MAX_FARE)]
record(f"4. Fare between ${MIN_FARE} and ${MAX_FARE}", df)

df = df[df["MktCoupons"] <= 3]
record("5. At most 2 connections", df)

df = df[~df["TkCarrier"].isin(["99", "99.0"])]
record("6. Drop interline (carrier = 99)", df)

# ---------- market and product ids ----------
lo = df[["OriginCityMarketID", "DestCityMarketID"]].min(axis=1).astype(int)
hi = df[["OriginCityMarketID", "DestCityMarketID"]].max(axis=1).astype(int)
keep = lo != hi                    # drop same-city-market records
df, lo, hi = df[keep].copy(), lo[keep], hi[keep]
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
record("7. Products before small-cell filter",
       prod.rename(columns={"sampled_pax": "Passengers"}))

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
