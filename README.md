# EC 560 Project, Checkpoint 1: Demand-estimation input data (US airlines)

Builds a pyBLP-ready product-market dataset from BTS DB1B Market data, 2019 Q1-Q4.

## Reproduce
    python -m venv .venv && source .venv/Scripts/activate
    python -m pip install -r requirements.txt
    bash run_all.sh 2>&1 | tee logs/run_log.txt

Raw files (about 100 MB each) are downloaded by run_all.sh from BTS TranStats
into data/raw/ and are not stored in this repo (size limits).

## Pipeline
- code/00_check_quarters.py: checks each raw file (quarter, columns).
- code/02_clean.py: sample filters and product-level aggregation (output/filter_counts.csv).
- code/03_build_pyblp_input.py: market size, shares, sensitivity table.
- code/04_logit_check.py, 04b_within_check.py: sanity regressions (not the final model).
- code/05_characteristics_instruments.py: characteristics and BLP instruments.

## Outputs
- data/clean/pyblp_input_2019_full.csv: final dataset for pyBLP.
- output/: filter counts, market-size sensitivity, summary statistics.
- logs/run_log.txt: log of the most recent complete run.
