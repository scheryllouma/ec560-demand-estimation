"""06_market_size.py: market size from OUTSIDE data (metro population) + income for heterogeneity.

Download these into data/outside/ (see the runbook):
  airports.csv         OurAirports airports file (needs iata_code, latitude_deg, longitude_deg,
                       municipality, iso_region, iso_country)
  cbsa_pop.csv         Census CBSA population estimates 2010-2019 (needs CBSA, NAME, LSAD, POPESTIMATE2019)
  cbsa_gazetteer.txt   Census 2019 Gazetteer, CBSA file, tab-delimited (needs CBSAFP or GEOID, INTPTLAT, INTPTLONG)
  acs_income.csv       OPTIONAL: median household income by CBSA (a cbsa column and a B19013_001E column)
  manual_crosswalk.csv OPTIONAL: your own fixes, columns city_market_id,cbsa

Matching airports to metro areas (CBSAs): (1) airport city name + state against the CBSA title;
(2) fallback: nearest CBSA centroid within MAX_KM. Everything not matched is listed for you to check.
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

PHI = None             # fraction of sqrt(pop_a*pop_b) that is the potential market. None = calibrate below.
CAL_QUANTILE = 0.999   # calibration: this quantile of the market-level (quantity / sqrt(pop_a*pop_b))
CAL_TARGET = 0.50      # ... gets an inside share of this value
MAX_INSIDE = 0.95      # floor: no market may have an inside share above this
MAX_KM = 100           # nearest-centroid fallback distance limit
OUT = "data/outside/"


def need(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. {what}")


def clean(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bft\b", "fort", s)
    s = re.sub(r"\bmt\b", "mount", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


# ---------- load ----------
prod = pd.read_csv("data/clean/products_2019.csv", dtype={"TkCarrier": str, "firm_ids": str, "product_ids": str})
need("data/clean/city_market_airports.csv", "Rerun code/02_clean.py first (it saves this file).")
xw = pd.read_csv("data/clean/city_market_airports.csv")

need(OUT + "airports.csv", "Download the OurAirports airports.csv.")
ap = pd.read_csv(OUT + "airports.csv", low_memory=False)
miss = [c for c in ["iata_code", "latitude_deg", "longitude_deg", "municipality", "iso_region", "iso_country"]
        if c not in ap.columns]
if miss:
    raise SystemExit(f"airports.csv is missing {miss}. Its columns are: {list(ap.columns)}")
ap = ap[(ap["iso_country"] == "US") & ap["iata_code"].notna()].drop_duplicates("iata_code")
ap["state"] = ap["iso_region"].astype(str).str.replace("US-", "", regex=False)

need(OUT + "cbsa_pop.csv", "Download the Census CBSA population file.")
pop = pd.read_csv(OUT + "cbsa_pop.csv", encoding="latin-1", low_memory=False)
miss = [c for c in ["CBSA", "NAME", "LSAD", "POPESTIMATE2019"] if c not in pop.columns]
if miss:
    raise SystemExit(f"cbsa_pop.csv is missing {miss}. Its columns are: {list(pop.columns)}")
pop = pop[pop["LSAD"].astype(str).str.contains("Statistical Area") & pop["MDIV"].isna()] if "MDIV" in pop.columns \
    else pop[pop["LSAD"].astype(str).str.contains("Statistical Area")]
pop = pop[["CBSA", "NAME", "POPESTIMATE2019"]].rename(columns={"CBSA": "cbsa", "NAME": "cbsa_name",
                                                              "POPESTIMATE2019": "pop2019"})
pop["cbsa"] = pop["cbsa"].astype(int)
pop = pop.drop_duplicates("cbsa")
print("CBSAs in population file:", len(pop))

# title lookup: (state, city token) -> list of cbsa
lookup = {}
for r in pop.itertuples():
    name = str(r.cbsa_name)
    if ", " not in name:
        continue
    cities, st = name.rsplit(", ", 1)
    toks = {clean(cities)} | {clean(t) for t in cities.split("-")}
    for s in st.split("-"):
        for t in toks:
            lookup.setdefault((s.strip(), t), []).append(r.cbsa)
popd = pop.set_index("cbsa")["pop2019"].to_dict()

# ---------- one airport per city market: the busiest one we have coordinates for ----------
m = xw.merge(ap[["iata_code", "latitude_deg", "longitude_deg", "municipality", "state"]],
             left_on="airport", right_on="iata_code", how="inner")
m = m.sort_values("Passengers", ascending=False).drop_duplicates("city_market_id")
cm = xw.groupby("city_market_id", as_index=False)["Passengers"].sum().rename(columns={"Passengers": "pax_total"})
cm = cm.merge(m[["city_market_id", "airport", "municipality", "state", "latitude_deg", "longitude_deg"]],
              on="city_market_id", how="left")

# ---------- match 1: city name + state ----------
def by_name(row):
    if pd.isna(row["municipality"]):
        return np.nan
    muni = clean(row["municipality"])
    cands = []
    for t in {muni} | set(clean(x) for x in re.split(r"[-/]", str(row["municipality"]))):
        cands += lookup.get((row["state"], t), [])
    return max(cands, key=lambda c: popd.get(c, 0)) if cands else np.nan

cm["cbsa"] = cm.apply(by_name, axis=1)
cm["match_type"] = np.where(cm["cbsa"].notna(), "name", "")

# ---------- match 2: nearest CBSA centroid ----------
if os.path.exists(OUT + "cbsa_gazetteer.txt"):
    gz = pd.read_csv(OUT + "cbsa_gazetteer.txt", sep="\t", dtype=str)
    gz.columns = [c.strip() for c in gz.columns]
    code = "CBSAFP" if "CBSAFP" in gz.columns else "GEOID"
    gz = gz.rename(columns={code: "cbsa"})
    gz["cbsa"] = gz["cbsa"].astype(int)
    gz["lat"] = gz["INTPTLAT"].astype(float)
    gz["lon"] = gz["INTPTLONG"].astype(float)
    gz = gz[gz["cbsa"].isin(popd)]
    todo = cm["cbsa"].isna() & cm["latitude_deg"].notna()
    for i in cm.index[todo]:
        dkm = haversine(cm.at[i, "latitude_deg"], cm.at[i, "longitude_deg"], gz["lat"].values, gz["lon"].values)
        k = int(np.argmin(dkm))
        if dkm[k] <= MAX_KM:
            cm.at[i, "cbsa"] = gz["cbsa"].values[k]
            cm.at[i, "match_type"] = f"nearest ({dkm[k]:.0f} km)"
else:
    print("NOTE: no cbsa_gazetteer.txt, so no nearest-centroid fallback was used")

# ---------- manual fixes ----------
if os.path.exists(OUT + "manual_crosswalk.csv"):
    mf = pd.read_csv(OUT + "manual_crosswalk.csv")
    for r in mf.itertuples():
        cm.loc[cm["city_market_id"] == r.city_market_id, ["cbsa", "match_type"]] = [r.cbsa, "manual"]
    print("Applied", len(mf), "manual fixes")

cm["cbsa"] = pd.to_numeric(cm["cbsa"], errors="coerce")
cm["pop2019"] = cm["cbsa"].map(popd)
cm["cbsa_name"] = cm["cbsa"].map(pop.set_index("cbsa")["cbsa_name"].to_dict())
cm["match_type"] = cm["match_type"].replace("", "UNMATCHED")
cm.loc[cm["cbsa"].isna(), "match_type"] = "UNMATCHED"

tot = cm["pax_total"].sum()
print("\nCity markets:", len(cm))
print("Share of passengers (airport-endpoint) by match type:")
print((cm.groupby(cm["match_type"].str.replace(r" \(.*\)", "", regex=True))["pax_total"].sum() / tot).round(4))
review = cm[cm["match_type"].ne("name") & cm["match_type"].ne("manual")].sort_values("pax_total", ascending=False)
review.to_csv("output/market_size_review.csv", index=False)
cm.to_csv("output/market_size_crosswalk.csv", index=False)
print("\nCHECK THESE BY HAND (largest unmatched / nearest-centroid matches). Fix in data/outside/manual_crosswalk.csv:")
print(review[["city_market_id", "airport", "municipality", "state", "cbsa_name", "match_type", "pax_total"]]
      .head(25).to_string(index=False))
print("\nLargest 15 city markets and their matches (sanity check):")
print(cm.sort_values("pax_total", ascending=False)[["city_market_id", "airport", "municipality", "cbsa_name", "pop2019"]]
      .head(15).to_string(index=False))

# ---------- income (optional) ----------
inc = None
if os.path.exists(OUT + "acs_income.csv"):
    a = pd.read_csv(OUT + "acs_income.csv")
    ccol = next((c for c in a.columns if c.lower() in ("cbsa", "geoid", "metropolitan statistical area/micropolitan statistical area")), None)
    icol = next((c for c in a.columns if "B19013" in c or c.lower() in ("median_income", "income")), None)
    if ccol and icol:
        a["cbsa"] = pd.to_numeric(a[ccol].astype(str).str[-5:], errors="coerce")
        inc = pd.to_numeric(a[icol], errors="coerce").where(lambda s: s > 0)
        inc = pd.Series(inc.values, index=a["cbsa"].values).groupby(level=0).first().to_dict()
        print("\nIncome loaded for", len(inc), "CBSAs")
    else:
        print("\nWARNING: could not find cbsa/income columns in acs_income.csv. Columns:", list(a.columns))
cm["income"] = cm["cbsa"].map(inc) if inc else np.nan

# ---------- attach to products ----------
c = cm.set_index("city_market_id")
for side in ["a", "b"]:
    prod[f"cbsa_{side}"] = prod[f"city_{side}"].map(c["cbsa"])
    prod[f"pop_{side}"] = prod[f"city_{side}"].map(c["pop2019"])
    prod[f"inc_{side}"] = prod[f"city_{side}"].map(c["income"])
before = prod["quantity"].sum()
bad = prod["pop_a"].isna() | prod["pop_b"].isna()
print(f"\nDropping {bad.sum():,} products in {prod.loc[bad, 'market_ids'].nunique():,} markets with an "
      f"unmatched endpoint: {prod.loc[bad, 'quantity'].sum() / before:.2%} of quantity")
prod = prod[~bad].copy()
prod["inc_mean"] = prod[["inc_a", "inc_b"]].mean(axis=1)

# ---------- market size ----------
prod["mkt_quantity"] = prod.groupby("market_ids")["quantity"].transform("sum")
prod["gm_pop"] = np.sqrt(prod["pop_a"] * prod["pop_b"])
mk = prod.drop_duplicates("market_ids")
phi = PHI if PHI is not None else (mk["mkt_quantity"] / mk["gm_pop"]).quantile(CAL_QUANTILE) / CAL_TARGET
print(f"\nPHI (fraction of sqrt(pop_a x pop_b) in the market) = {phi:.4f}"
      f"{'  (calibrated)' if PHI is None else '  (set by you)'}")

def market_size(mult):
    return np.maximum(mult * phi * mk["gm_pop"], mk["mkt_quantity"] / MAX_INSIDE)

rows = []
for mult in [0.5, 1.0, 2.0]:
    ms = market_size(mult)
    ins = mk["mkt_quantity"] / ms
    rows.append({"phi": mult * phi, "mean_inside": ins.mean(), "median_inside": ins.median(),
                 "p99_inside": ins.quantile(0.99), "share_floor_binding": (mult * phi * mk["gm_pop"] < mk["mkt_quantity"] / MAX_INSIDE).mean()})
sens = pd.DataFrame(rows)
print("\nSensitivity of inside share to PHI:\n", sens.round(4).to_string(index=False))
sens.to_csv("output/market_size_sensitivity.csv", index=False)

prod["market_size"] = np.maximum(phi * prod["gm_pop"], prod["mkt_quantity"] / MAX_INSIDE)
print("Markets where the 95% floor binds:", f"{(phi * mk['gm_pop'] < mk['mkt_quantity'] / MAX_INSIDE).mean():.2%}")
prod["shares"] = prod["quantity"] / prod["market_size"]
assert prod["shares"].gt(0).all() and prod.groupby("market_ids")["shares"].sum().lt(1).all()
prod.to_csv("data/clean/products_2019_ms.csv", index=False)
print("\nSaved data/clean/products_2019_ms.csv:", f"{len(prod):,}", "products,", f"{prod['market_ids'].nunique():,}", "markets")
print("Inside share by market:\n", prod.groupby("market_ids")["shares"].sum().describe().round(4))
