import os
import zipfile
import pandas as pd

for q in [1, 2, 3, 4]:
    path = f"data/raw/db1b_market_2019_q{q}.zip"
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nFile q{q}: {size_mb:.1f} MB")
    try:
        with zipfile.ZipFile(path) as z:
            print("  contains:", z.namelist())
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as fh:
                d = pd.read_csv(fh, nrows=2)
        print("  n columns:", d.shape[1], "| has state cols:",
              any("state" in c.lower() for c in d.columns))
        print("  year/quarter:", d.iloc[0].filter(regex="(?i)^(year|quarter)$").to_dict())
    except Exception as e:
        print("  PROBLEM:", type(e).__name__, e)
        with open(path, "rb") as fh:
            print("  first bytes:", fh.read(60))
