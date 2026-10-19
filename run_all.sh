#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw data/clean output logs
date
python --version
python -m pip freeze | grep -iE "^(pandas|numpy|pyblp)=="

# Step 0: download raw data (skips files already present)
for q in 1 2 3 4; do
  f=data/raw/db1b_market_2019_q$q.zip
  if [ ! -s "$f" ]; then
    curl -L --fail -o "$f" \
      "https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_2019_$q.zip"
  fi
done

python -u code/00_check_quarters.py
python -u code/02_clean.py
python -u code/03_build_pyblp_input.py
python -u code/04_logit_check.py
python -u code/04b_within_check.py
python -u code/05_characteristics_instruments.py
echo "DONE: $(date)"
