"""02_clean.py (v2): filter DB1B Market data, pool small products into a fringe, build product data.

Market  = undirected city-market pair x quarter
Product = ticketing carrier x {nonstop, connecting} within a market.
          Products with fewer than MIN_SAMPLED_PAX sampled passengers are pooled
          into ONE 'FRINGE' product per market instead of being dropped.
"""
import glob
import re
import zipfile
import pandas as pd

SAMPLE_SCALE = 10       # DB1B is a 10% sample in 2019
MIN_FARE, MAX_FARE = 25, 2000
MIN_SAMPLED_PAX = 20    # products below this are pooled into the fringe
MIN_MARKET_PAX = 20     # markets with fewer total sampled passengers are dropped


def read_member(path, **kw):
    """Read the CSV inside a BTS zip (some zips also contain readme.html)."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as fh:
            return pd.read_csv(fh, **kw)


def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


ALIAS = {
    "year": "Year", "quarter": "Quarter",
    "marketcoupons": "MktCoupons", "mktcoupons": "MktCoupons",
    "origincitymarketid": "OriginCityMarketID", "destcitymarketid": "DestCityMarketID",
    "origin": "Origin", "dest": "Dest",
    "originstate": "OriginState", "originstateabr": "OriginState",
    "deststate": "DestState", "deststateabr": "DestState",
    "ticketcarrier": "TkCarrier", "tkcarrier": "TkCarrier",
    "bulkfare": "BulkFare", "passengers": "Passengers",
    "marketfare": "MktFare", "mktfare": "MktFare",
    "marketdistance": "MktDistance", "mktdistance": "MktDistance",
}
REQUIRED = ["Year", "Quarter", "MktCoupons", "OriginCityMarketID", "DestCityMarketID",
            "TkCarrier", "BulkFare", "Passengers", "MktFare", "MktDistance"]

# ---------- load all quarters ----------
files = sorted(glob.glob("data/raw/db1b_market_2019_q*.zip"))
print("Files found:", files)
if not files:
    raise SystemExit("No files in data/raw/ matching db1b_market_2019_q*.zip")
parts = []
for f in files:
    print("File columns:", list(read_member(f, nrows=0).columns))
    d = read_member(f, usecols=lambda c: norm(c) in ALIAS, low_memory=False)
    parts.append(d.rename(columns={c: ALIAS[norm(c)] for c in d.columns}))
df = pd.concat(parts, ignore_index=True)
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}. Send me the 'File columns' lines.")
df["TkCarrier"] = df["TkCarrier"].astype(str).str.strip()

# ---------- filter bookkeeping ----------
log = []
def record(step, rows, pax):
    log.append({"step": step, "rows": rows, "passengers": pax})
    print(f"{step:<52} rows={rows:>12,}  pax={pax:>14,.0f}")
def rec_df(step, d, pax_col="Passengers"):
    record(step, len(d), d[pax_col].sum())

rec_df("0. Raw data", df)
df = df.dropna(subset=["MktFare", "Passengers", "TkCarrier", "OriginCityMarketID", "DestCityMarketID"])
rec_df("1. Drop missing key fields", df)
if "OriginState" in df.columns and "DestState" in df.columns:
    out = ["AK", "HI", "PR", "VI", "TT", "GU", "MP", "AS"]
    df = df[~df["OriginState"].isin(out) & ~df["DestState"].isin(out)]
    rec_df("2. Continental US only", df)
df = df[df["BulkFare"] == 0]
rec_df("3. Drop bulk fares", df)
df = df[(df["MktFare"] >= MIN_FARE) & (df["MktFare"] <= MAX_FARE)]
rec_df(f"4. Fare between ${MIN_FARE} and ${MAX_FARE}", df)
df = df[df["MktCoupons"] <= 3]
rec_df("5. At most 2 connections", df)
df = df[~df["TkCarrier"].isin(["99", "99.0"])]
rec_df("6. Drop interline (carrier = 99)", df)

# ---------- airport crosswalk for outside-data matching ----------
if "Origin" in df.columns and "Dest" in df.columns:
    a = df[["OriginCityMarketID", "Origin", "Passengers"]]
    a.columns = ["city_market_id", "airport", "Passengers"]
    b = df[["DestCityMarketID", "Dest", "Passengers"]]
    b.columns = ["city_market_id", "airport", "Passengers"]
    xw = pd.concat([a, b]).groupby(["city_market_id", "airport"], as_index=False)["Passengers"].sum()
    xw.to_csv("data/clean/city_market_airports.csv", index=False)
    print("Saved data/clean/city_market_airports.csv:", len(xw), "city-market/airport pairs")
else:
    print("WARNING: no Origin/Dest airport columns, so no airport crosswalk was saved")

# ---------- market and product ids ----------
lo = df[["OriginCityMarketID", "DestCityMarketID"]].min(axis=1).astype(int)
hi = df[["OriginCityMarketID", "DestCityMarketID"]].max(axis=1).astype(int)
keep = lo != hi
df, lo, hi = df[keep].copy(), lo[keep], hi[keep]
df["city_a"], df["city_b"] = lo.astype(int), hi.astype(int)
df["route"] = lo.astype(str) + "_" + hi.astype(str)
df["quarter"] = df["Year"].astype(int).astype(str) + "Q" + df["Quarter"].astype(int).astype(str)
df["market_ids"] = df["route"] + "_" + df["quarter"]
df["nonstop"] = (df["MktCoupons"] == 1).astype(int)
df["fare_x_pax"] = df["MktFare"] * df["Passengers"]
df["dist_x_pax"] = df["MktDistance"] * df["Passengers"]

# ---------- aggregate to product level ----------
prod = (df.groupby(["market_ids", "route", "quarter", "city_a", "city_b", "TkCarrier", "nonstop"],
                   as_index=False)
          .agg(fare_x_pax=("fare_x_pax", "sum"), dist_x_pax=("dist_x_pax", "sum"),
               sampled_pax=("Passengers", "sum")))
record("7. Products before pooling (products)", len(prod), prod["sampled_pax"].sum())

# ---------- pool small products into one fringe product per market ----------
small = prod["sampled_pax"] < MIN_SAMPLED_PAX
big, sm = prod[~small].copy(), prod[small].copy()
big["n_pooled"] = 0
big["product_ids"] = big["TkCarrier"] + "_" + big["nonstop"].astype(str)
sm["nonstop_x_pax"] = sm["nonstop"] * sm["sampled_pax"]
fr = (sm.groupby(["market_ids", "route", "quarter", "city_a", "city_b"], as_index=False)
        .agg(fare_x_pax=("fare_x_pax", "sum"), dist_x_pax=("dist_x_pax", "sum"),
             sampled_pax=("sampled_pax", "sum"), nonstop_x_pax=("nonstop_x_pax", "sum"),
             n_pooled=("TkCarrier", "size")))
fr["TkCarrier"] = "FRINGE"
fr["nonstop"] = fr["nonstop_x_pax"] / fr["sampled_pax"]      # share of fringe passengers flying nonstop
fr["product_ids"] = "FRINGE"
fr = fr.drop(columns="nonstop_x_pax")
print(f"Pooled {int(small.sum()):,} small products into {len(fr):,} fringe products "
      f"({sm['sampled_pax'].sum() / prod['sampled_pax'].sum():.1%} of sampled passengers)")
prod = pd.concat([big, fr], ignore_index=True)
record("8. After pooling small products into fringe (products)", len(prod), prod["sampled_pax"].sum())

mk_pax = prod.groupby("market_ids")["sampled_pax"].transform("sum")
prod = prod[mk_pax >= MIN_MARKET_PAX].copy()
record(f"9. Markets with >= {MIN_MARKET_PAX} sampled pax (products)", len(prod), prod["sampled_pax"].sum())

prod["prices"] = prod["fare_x_pax"] / prod["sampled_pax"]
prod["distance"] = prod["dist_x_pax"] / prod["sampled_pax"]
prod["quantity"] = prod["sampled_pax"] * SAMPLE_SCALE
prod["firm_ids"] = prod["TkCarrier"]
prod = prod.drop(columns=["fare_x_pax", "dist_x_pax"])

pd.DataFrame(log).to_csv("output/filter_counts.csv", index=False)
prod.to_csv("data/clean/products_2019.csv", index=False)
print("\nProducts:", f"{len(prod):,}", "| Markets:", f"{prod['market_ids'].nunique():,}",
      "| Routes:", f"{prod['route'].nunique():,}")
print("Fringe products:", f"{(prod['firm_ids'] == 'FRINGE').sum():,}")
print("Products per market:\n", prod.groupby("market_ids").size().describe())
print("\nPrice summary:\n", prod["prices"].describe())
