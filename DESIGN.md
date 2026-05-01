# VigiHealth Design

## Goal

Build a local-first global health analytics platform on WHO GHO data, using the same medallion-lake pattern as the F1 project, but adapted for country-year epidemiology and geographic visualization.

## Core principles

- Annual cadence only, unless a future data source explicitly adds higher frequency.
- Country-year is the unit of analysis.
- ISO3 country codes everywhere.
- Time-based splits only.
- Keep heavy imports out of module import time.
- Make the first version small enough to ship and test.

## Proposed architecture

```text
WHO GHO OData API
  -> raw Parquet per indicator
  -> DuckDB bronze panel
  -> wide temporal silver feature store
  -> target-specific gold ABTs
  -> ML training / evaluation
  -> Streamlit app with charts and maps
```

### Repository layout

- `constants.py` — central `pathlib.Path` definitions
- `etl/collect.py` — WHO GHO ingestion and raw indicator export
- `etl/bronze.py` — tidy indicator panel
- `etl/silver.py` — country temporal features
- `etl/gold.py` — ABT construction for each target
- `etl/run_pipeline.py` — orchestration
- `ml/model_selection.py` — candidate model registry
- `ml/outbreak_model.py` — target 1 training
- `ml/resurgence_model.py` — target 2 training
- `ml/vaccination_model.py` — target 3 training
- `ml/predict.py` — loading models and generating predictions
- `app/main.py` — Streamlit wiring only
- `app/tab_predictions.py` — predictions UI
- `app/tab_model_comparison.py` — metrics and plots
- `app/tab_eda.py` — exploration
- `app/tab_world_map.py` — choropleths and animated maps
- `app/tab_duckdb.py` — SQL console
- `app/helpers.py` — shared UI helpers

## Data model

### Raw
One Parquet per indicator code.

### Bronze
A long tidy panel with at least:
- `country_iso3`
- `year`
- `indicator_code`
- `value`
- any available dimensions carried through explicitly

### Silver
Country-year wide feature tables with temporal windows:
- lifetime
- last 5 years
- last 10 years
- last 20 years
- all years joined together for modeling convenience

### Gold
One ABT per target, with explicit leakage exclusions documented in code comments.

## Target framing

### Target 1 — outbreak risk forecast
Binary forecast of whether a country and disease will exceed its recent historical trend.

### Target 2 — disease resurgence detection
Binary classification of trend reversal after sustained improvement.

### Target 3 — vaccination coverage milestone
Binary forecast of whether a vaccine coverage threshold will be reached inside a chosen horizon.

### Target 4 — healthcare strain
Deferred to v2.

## Implementation order

### Phase 0 — project scaffold
- Add `constants.py`
- Add package `__init__.py` files
- Add `.env.example`
- Add minimal app and pipeline entrypoints

### Phase 1 — ingestion
- implement WHO GHO catalog loading
- implement paginated indicator fetch
- write raw Parquet files
- add retry/backoff and idempotency

### Phase 2 — bronze
- normalize indicator rows into a tidy panel
- filter regional aggregates out of model training data
- keep schema stable and long

### Phase 3 — silver
- compute lags, rolling means, rolling slopes
- keep all features point-in-time correct
- export one Parquet per window

### Phase 4 — gold
- create one ABT per target
- define target labels explicitly
- audit leakage columns in comments

### Phase 5 — ML
- register candidate models
- use time-based splits
- log PR-AUC first, ROC-AUC second
- keep MLflow artifacts simple and reproducible

### Phase 6 — app
- keep `main.py` as layout-only wiring
- put all logic in tab modules
- cache Parquet reads
- add map tab last, after the data model stabilizes

## TDD bootstrap order

Start by making these tests pass first:
1. path/constants contract
2. module importability
3. helper API contract
4. ETL orchestration contract
5. model registry contract
6. app render-function contract

Then implement one slice at a time, keeping the suite green.

## Notes for tomorrow's test run

If pytest fails on imports, that is expected at this stage. The first implementation step should be creating the package scaffold and constants module, not jumping straight into ingestion.
