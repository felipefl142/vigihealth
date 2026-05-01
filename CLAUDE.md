# CLAUDE.md — VigiHealth

Guidance for Claude Code working in this repository. Read this before making changes.

## Project Summary

**VigiHealth** is a multi-target global health analytics platform. It ingests WHO Global Health Observatory (GHO) indicators via the OData API, builds a medallion lake on local Parquet (DuckDB engine), trains multiple ML models per prediction target tracked in MLFlow, and serves predictions through a Streamlit app with geographic visualizations.

This is a sister project to `f1-predict-analysis-platform` and intentionally reuses its architecture. When in doubt about a pattern, check that repo first.

## Architecture

```
WHO GHO OData API → data/raw/ → DuckDB → data/{bronze,silver,gold}/ → ML Models → Streamlit
```

Medallion layers:

| Layer | Path | Contents |
|---|---|---|
| Raw | `data/raw/` | One Parquet per indicator code, e.g. `WHOSIS_000001.parquet` |
| Bronze | `data/bronze/` | Cleaned country × year × indicator panel |
| Silver | `data/silver/` | Per-country temporal feature stores (`fs_country_life`, `fs_country_last5`, `fs_country_last10`, `fs_country_last20`, `fs_country_all`) |
| Gold | `data/gold/` | One ABT per ML target |

## ML Targets

| Target | Status | ABT | Model module |
|---|---|---|---|
| Outbreak risk forecast | v1 | `abt_outbreak_risk` | `ml/outbreak_model.py` |
| Disease resurgence detection | v1 | `abt_resurgence` | `ml/resurgence_model.py` |
| Vaccination coverage milestone | v1 | `abt_vaccination_milestone` | `ml/vaccination_model.py` |
| Healthcare system strain | v2 (deferred) | `abt_capacity_strain` | `ml/capacity_model.py` |

See `Plan.md` for full target framings and rationale.

## Tech Stack

- **Python 3.12+**
- **Data ingestion:** `requests` against `https://ghoapi.azureedge.net/api/`
- **Storage:** Parquet (pyarrow)
- **Query engine:** DuckDB
- **ML:** scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Tuning:** Optuna (TPE sampler, median pruner)
- **Tracking:** MLFlow with SQLite backend
- **App:** Streamlit
- **Charts:** Plotly, Matplotlib
- **Maps:** Folium (choropleths), pydeck (time-series animations)
- **Containers:** Docker + docker-compose

## Repository Layout

```
app/                  # Streamlit web app
  main.py
  tab_predictions.py
  tab_model_comparison.py
  tab_eda.py
  tab_world_map.py    # NEW vs F1 project
  tab_duckdb.py
  helpers.py
etl/                  # Pipeline modules
  collect.py          # GHO OData ingestion
  bronze.py
  silver.py
  gold.py
  run_pipeline.py
  sql/                # DuckDB SQL queries (one file per view/ABT)
ml/                   # Per-target training + utilities
data/                 # Parquet files (gitignored)
mlruns/               # MLFlow artifacts (gitignored)
mlflow.db             # MLFlow metadata (gitignored)
notebooks/            # Exploration only — not part of the pipeline
```

## Running the Pipeline

```bash
# Full pipeline — collects, then builds all layers
python -m etl.run_pipeline --years 2000 2024

# Force re-collection of existing indicators
python -m etl.run_pipeline --years 2000 2024 --force

# Individual steps
python -m etl.collect --indicators WHS4_100 WHS4_544
python -m etl.bronze
python -m etl.silver
python -m etl.gold

# Train models
python -m ml.outbreak_model
python -m ml.resurgence_model
python -m ml.vaccination_model

# Streamlit app
streamlit run app/main.py
# MLFlow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Conventions Claude Should Follow

### Data layer

- **Country codes are ISO3.** GHO uses ISO3 natively. Always join on the ISO3 string. Handle regional aggregates (`SEAR`, `EUR`, `AFR`, etc.) by filtering them out before training — they leak signal into country predictions.
- **Time column is `year` (int).** GHO returns `TimeDimensionBegin` as a year integer for most indicators. If you encounter date ranges, use `TimeDimensionBegin` and document the choice in a code comment.
- **One Parquet per indicator in raw.** Filename = `{IndicatorCode}.parquet`. Do not consolidate at the raw layer.
- **Bronze is tidy long format:** columns `country_iso3`, `year`, `indicator_code`, `value`, `dim1`, `dim2`. Pivots happen in silver, not bronze.
- **Silver is wide.** One row per country–year, columns are derived feature names like `tb_incidence_rate_lag1`, `dtp3_coverage_trend5`.

### SQL

- **All SQL lives in `etl/sql/`** as standalone `.sql` files, loaded by Python at runtime. Mirrors the F1 project's pattern.
- Use DuckDB's `union_by_name` when stacking indicator files with mixed schemas.
- Window functions for trend computation; do not compute trends in pandas if SQL can do it.
- Comment any column that is excluded for leakage reasons — recruiters and reviewers look for this.

### ML

- **Time-based train/test splits, always.** Never use random splits. Default split: train ≤ 2018, test 2019+. Document the split year in MLFlow.
- **Primary metric: PR-AUC.** Targets are imbalanced; ROC-AUC alone is misleading. Always log both.
- **Candidate models per target:** LogisticRegression, BalancedRandomForest, LightGBM, XGBoost. Use `--nologreg` flag to skip LogisticRegression in CI runs.
- **Optuna budget:** 50 trials default for development, 200+ for final runs. Always use median pruner and TPE sampler.
- **MLFlow logging:** model, params, metrics (PR-AUC, ROC-AUC, F1, Brier), per-class confusion matrix, feature importance plot, and the country-year holdout predictions as an artifact CSV.

### App

- **Tabs are independent modules.** Each tab is its own Python file with a single `render()` function. `main.py` only wires tabs to layout.
- **Cache loaders.** Use `@st.cache_data` on every Parquet read in the app — do not re-read on every interaction.
- **Geographic viz:** prefer Folium for static choropleths (better legend control), pydeck for time-slider animations. Both should read from the same gold-layer Parquet.

### Code style

- **Type hints on public functions.** Internal helpers can skip them.
- **Logging via `logging` module**, not `print`. ETL scripts log INFO by default, DEBUG with `--verbose`.
- **No hardcoded paths.** Use `pathlib.Path` from a single `constants.py` defining `DATA_DIR`, `RAW_DIR`, etc.
- **`.env.example` is canonical** — any env var used in code must appear in `.env.example` with a comment.

## Things to Avoid

- Do **not** introduce monthly granularity for indicators that GHO publishes annually. Interpolated months are not real data and will silently corrupt every downstream metric.
- Do **not** add COVID-19 as a target without a separate data source — GHO explicitly does not publish COVID data.
- Do **not** mix random and time-based splits across targets. Pick time-based for all and stick with it.
- Do **not** use Streamlit caching with mutable state. Cache reads, not transformations that take user input.
- Do **not** commit `data/`, `mlruns/`, or `mlflow.db` — all are gitignored.

## When Asked to Add a New Indicator

1. Look it up at `https://ghoapi.azureedge.net/api/Indicator` to confirm the code exists
2. Add the code to the indicator config (typically in `etl/collect.py` or a YAML config)
3. Run `python -m etl.collect --indicators NEW_CODE` to fetch
4. Re-run bronze, silver, and the affected gold ABTs
5. Update the relevant ABT SQL to include the new feature
6. Re-train models — the new feature should appear in MLFlow feature importance plots

## When Asked to Add a New ML Target

1. Define the target precisely in `Plan.md` first (label definition, horizon, granularity)
2. Write an audit of which silver features could leak the label and exclude them explicitly
3. Create `etl/sql/abt_<target>.sql`
4. Create `ml/<target>_model.py` mirroring an existing target module's structure
5. Add a tab or extend `tab_predictions.py` to surface the new target
6. Document the target in `README.md` and this file

## References

- WHO GHO OData docs: https://www.who.int/data/gho/info/gho-odata-api
- GHO indicator browser: https://www.who.int/data/gho/indicator-metadata-registry
- Sister project: `f1-predict-analysis-platform` — same architecture, different domain
- Inspiration for medallion-on-Parquet pattern: TeoMeWhy/f1-lake
