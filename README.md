# VigiHealth

Multi-target global health analytics platform built on WHO Global Health Observatory (GHO) data. Ingests indicators via the OData API, builds a medallion lake on local Parquet (DuckDB engine), trains per-target ML models tracked in MLflow, and serves predictions and geographic visualizations through a Streamlit app.

## Architecture

```
WHO GHO OData API → data/raw/ → DuckDB → data/{bronze,silver,gold}/ → ML models → Streamlit
```

| Layer | Path | Contents |
|---|---|---|
| Raw | `data/raw/` | One Parquet per indicator code (e.g. `WHOSIS_000001.parquet`) |
| Bronze | `data/bronze/` | Cleaned country × year × indicator panel (long, tidy) |
| Silver | `data/silver/` | Per-country temporal feature stores (`fs_country_life`, `fs_country_last5/10/20`, `fs_country_all`) |
| Gold | `data/gold/` | One ABT per ML target |

## ML Targets

| Target | Status | ABT | Module |
|---|---|---|---|
| Outbreak risk forecast | v1 | `abt_outbreak_risk` | `ml/outbreak_model.py` |
| Disease resurgence detection | v1 | `abt_resurgence` | `ml/resurgence_model.py` |
| Vaccination coverage milestone | v1 | `abt_vaccination_milestone` | `ml/vaccination_model.py` |
| Healthcare system strain | v2 (deferred) | `abt_capacity_strain` | `ml/capacity_model.py` |

All targets use **time-based splits** (default: train ≤ 2018, test ≥ 2019) and report **PR-AUC** as the primary metric (ROC-AUC, F1, Brier logged alongside).

## Tech Stack

- Python 3.12+
- Ingestion: `requests` against `https://ghoapi.azureedge.net/api/`
- Storage: Parquet (pyarrow)
- Query engine: DuckDB
- ML: scikit-learn, XGBoost, LightGBM, imbalanced-learn
- Tuning: Optuna (TPE sampler, median pruner)
- Tracking: MLflow with SQLite backend
- App: Streamlit
- Charts: Plotly, Matplotlib
- Maps: Folium (choropleths), pydeck (time-series animations)

## Setup

```bash
git clone <repo-url> vigihealth
cd vigihealth

python -m venv .venv
source .venv/bin/activate          # bash/zsh
# source .venv/bin/activate.fish   # fish

pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` if it exists and adjust paths.

## Running the Pipeline

```bash
# Full pipeline — collect, then build all medallion layers
python -m etl.run_pipeline --years 2000 2024

# Force re-collection of indicators already on disk
python -m etl.run_pipeline --years 2000 2024 --force

# Individual steps
python -m etl.collect --indicators WHS4_100 WHS4_544
python -m etl.bronze
python -m etl.silver
python -m etl.gold
```

## Training Models

```bash
python -m ml.outbreak_model
python -m ml.resurgence_model
python -m ml.vaccination_model

# Skip LogisticRegression in CI / quick runs
python -m ml.outbreak_model --nologreg
```

Each run logs model, params, metrics (PR-AUC, ROC-AUC, F1, Brier), confusion matrix, feature importance, and holdout predictions to MLflow.

## Streamlit App

```bash
streamlit run app/main.py
```

Tabs:
- **Predictions** — per-target country predictions
- **Model Comparison** — metrics and plots across runs
- **EDA** — exploratory views of bronze/silver layers
- **World Map** — Folium choropleths and pydeck time-slider animations
- **DuckDB** — SQL console over the medallion lake

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://127.0.0.1:5000>.

## Repository Layout

```
app/                  # Streamlit web app (one file per tab)
etl/                  # Ingestion + medallion build
  sql/                # Standalone .sql files loaded at runtime
ml/                   # Per-target training + model registry
data/                 # Parquet files (gitignored)
mlruns/               # MLflow artifacts (gitignored)
mlflow.db             # MLflow metadata (gitignored)
tests/                # Pytest contract tests
notebooks/            # Exploration only (not part of the pipeline)
constants.py          # Central pathlib.Path definitions
CLAUDE.md             # Conventions for Claude Code working in this repo
DESIGN.md             # Architecture notes and implementation order
```

## Tests

```bash
pytest
```

## Conventions

- Country codes are **ISO3**; regional aggregates (`SEAR`, `EUR`, `AFR`, …) are filtered out before training.
- Time column is **`year` (int)**.
- One Parquet per indicator in raw — no consolidation at the raw layer.
- Bronze is long/tidy; pivots happen in silver, not bronze.
- All SQL lives in `etl/sql/` as standalone files loaded by Python at runtime.
- No hardcoded paths — everything goes through `constants.py`.
- Logging via the `logging` module, not `print`.

See `CLAUDE.md` for the full convention list and `DESIGN.md` for architectural rationale.

## References

- WHO GHO OData docs — <https://www.who.int/data/gho/info/gho-odata-api>
- GHO indicator browser — <https://www.who.int/data/gho/indicator-metadata-registry>
- Sister project (same medallion pattern, different domain): `f1-predict-analysis-platform`

## License

TBD.
