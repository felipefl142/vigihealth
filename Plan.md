# VigiHealth — Project Plan

A multi-target global health analytics platform built on WHO Global Health Observatory (GHO) data. Mirrors the architecture of `f1-predict-analysis-platform`: a free, local-first medallion lake (DuckDB + Parquet) feeding multiple ML models tracked in MLFlow, served through a Streamlit web app — extended with geographic visualizations.

## Why VigiHealth

The name (Vigil + Health) signals continuous surveillance, which is the core epidemiological framing: detect, monitor, anticipate. The project demonstrates the same data engineering and ML methodology applied in F1 transferred to a domain with social impact and very different data characteristics — annual cadence, country-level entities, sparse and uneven indicator coverage.

## Architecture

```
WHO GHO OData API
      │
      ▼
data/raw/        Parquet per indicator (one file per IndicatorCode)
      │  DuckDB
      ▼
data/bronze/     Cleaned and consolidated indicator panel (country × year × indicator)
      │
      ▼
data/silver/     Feature store — temporal windows per country (life, last 5y, last 10y, last 20y)
      │
      ▼
data/gold/       Analytical Base Tables (one per ML target)
      │
      ▼
ML Models  ──►  MLFlow  ──►  Streamlit App (with geographic viz)
```

Same medallion pattern as the F1 project. Key differences:

| F1 project | VigiHealth |
|---|---|
| FastF1 (sessions, weather) | WHO GHO OData (1000+ indicators) |
| Race-level granularity | Country–year granularity |
| ~24 events × ~20 drivers per year | ~245 countries × ~25 years |
| In-season vs end-of-year ABTs | Forecast horizon variants (1y, 3y, 5y) |
| 3 targets (champion / team / departure) | 4 targets (see below) |

## Data Source — WHO GHO

**Endpoint:** `https://ghoapi.azureedge.net/api/`

**Key facts that shape the project:**

- ~2300 indicators across 245 countries/regions, decades of history
- Mostly **annual** cadence — *not monthly* (this constrains target framing, see below)
- No authentication, no rate limits documented (be polite anyway)
- OData filtering supported (`$filter=SpatialDim eq 'BRA'`, `TimeDimensionBegin`, etc.)
- API does **not** provide COVID-19 data — that lives on UNOCHA HDX as CSVs
- Indicator metadata at `/api/Indicator`, dimension values at `/api/DIMENSION/{name}/DimensionValues`

**Optional secondary sources** (introduced in later phases, not v1):

- WHO Disease Outbreak News — scraped for higher-frequency outbreak signals
- Our World in Data (CSVs on GitHub) — supplements where GHO is sparse, especially for vaccination
- World Bank indicators — for economic and health-system covariates (GDP, health expenditure)

## ML Targets

You picked four targets. Building all four at once is more scope than the F1 project's three. The plan below treats targets 1–3 as **v1 (must-ship)** and target 4 as **v2 (deferred)** to keep the first release tight. All four are documented so the v2 work is already scoped.

> **Reality check on the original target framing:** GHO's annual cadence and lack of real-time hospital data force the original phrasings to be reframed. The reframings below preserve the spirit of each target while staying within what the data can actually support. If you later add OWID or WHO Disease Outbreak News, the original framings become reachable.

### Target 1 — Outbreak Risk Forecast (v1)

**Original framing:** "Outbreak risk by country-month."
**Reframed:** *Will country C experience an above-trend incidence year for disease D in the next reporting year?*

- Disease scope for v1: tuberculosis (`TB_*` indicators), malaria (`MALARIA_*`), measles (vaccination + case indicators)
- Label: binary — incidence rate exceeds the country's rolling 5-year median by >1 standard deviation
- Granularity: country–year (not country–month) — honest about GHO's cadence
- Predictors: lagged incidence, vaccination coverage, health expenditure, water/sanitation access, population density, neighbouring-country incidence
- Sanity check: this is essentially "anomaly classification on a sparse panel," which is well-trodden ground

### Target 2 — Disease Resurgence Detection (v1)

*Will a disease that has been declining in country C for ≥5 years reverse trend in the next 1–3 years?*

- Same disease scope as Target 1, plus polio and pertussis
- Label: binary — reversal of a multi-year declining trend (operationalized as: rolling 3-year mean of incidence rises above the prior 5-year mean)
- Predictors: trend slope, vaccination coverage trajectory, conflict/displacement proxies (where available), neighbouring resurgence
- This is the cleanest "interesting headline" target — resurgence after eradication progress is policy-relevant and underexplored

### Target 3 — Vaccination Coverage Milestone (v1)

*Will country C reach ≥90% coverage for vaccine V by year Y?*

- Vaccine scope: DTP3, MCV1 (measles), HepB3, polio
- Indicators: `WHS4_*` immunization series
- Label: binary — milestone achieved within the forecast horizon (1y / 3y / 5y variants)
- Predictors: current coverage, 5-year coverage trajectory, health expenditure per capita, conflict indicators, urbanization
- Closest analog to F1's "champion" target — clear binary outcome, well-defined horizon

### Target 4 — Healthcare System Strain (v2, deferred)

**Original framing:** "ICU/bed capacity breach."
**Reframed:** *Will country C's hospital bed density fall below the disease-burden-adjusted threshold within the next 5 years?*

- Indicator: hospital bed density per 10,000 population (annual, structural — not real-time)
- Label: binary — capacity falls below a derived threshold given the country's NCD + communicable disease burden
- Why deferred: the indicator is sparser than the others, threshold derivation needs careful methodology, and three targets is enough to ship a strong v1
- When ready: revisit with World Bank health system covariates as supplementary inputs

## Tech Stack

Identical to the F1 project, **plus** geographic visualization:

| Layer | Tool |
|---|---|
| Data collection | `requests` against the GHO OData endpoint, optionally `ghoclient` |
| Storage | Local Parquet files |
| SQL engine | DuckDB |
| Data processing | pandas |
| ML models | scikit-learn, XGBoost, LightGBM |
| Class balancing | imbalanced-learn |
| Hyperparameter tuning | Optuna |
| Forecasting (optional) | Prophet or TimesFM zero-shot, in a separate venv |
| Experiment tracking | MLFlow with SQLite backend |
| Web app | Streamlit |
| Charts | Plotly, Matplotlib |
| **Maps (new)** | Folium for country choropleths, pydeck for animated time-series maps |
| Containerization | Docker + docker-compose |

## Project Structure

```
vigihealth/
├── app/
│   ├── main.py                       # Entry point
│   ├── tab_predictions.py            # ML predictions per target
│   ├── tab_model_comparison.py       # ROC, PR, confusion matrices
│   ├── tab_eda.py                    # Exploratory analysis
│   ├── tab_world_map.py              # NEW — choropleth + animated time-series map
│   ├── tab_duckdb.py                 # SQL console
│   └── helpers.py
├── etl/
│   ├── collect.py                    # GHO OData ingestion
│   ├── bronze.py                     # Raw → cleaned indicator panel
│   ├── silver.py                     # Bronze → temporal feature store
│   ├── gold.py                       # Silver → ABTs per target
│   ├── run_pipeline.py
│   └── sql/
│       ├── fs_country.sql                    # Per-country temporal windows
│       ├── fs_all.sql                        # Join temporal windows
│       ├── abt_outbreak_risk.sql
│       ├── abt_resurgence.sql
│       ├── abt_vaccination_milestone.sql
│       └── abt_capacity_strain.sql           # v2
├── ml/
│   ├── outbreak_model.py
│   ├── resurgence_model.py
│   ├── vaccination_model.py
│   ├── capacity_model.py             # v2
│   ├── model_selection.py
│   ├── predict.py
│   ├── utils.py
│   └── evaluate_timesfm.py           # optional, for forecast comparison
├── data/                             # raw / bronze / silver / gold
├── mlruns/
├── notebooks/                        # exploration
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── README.md
├── Plan.md
├── CLAUDE.md
└── .env.example
```

## Phased Roadmap

### Phase 0 — Scaffold (1–2 days)

- Repo, `requirements.txt`, `Dockerfile`, `docker-compose.yaml`, basic Streamlit "hello world", MLFlow up
- Decision log started in `notebooks/00_indicator_scoping.ipynb`

### Phase 1 — Indicator scoping (2–3 days)

- Pull `/api/Indicator` and inventory the ~2300 indicators
- For each ML target, finalize the shortlist of predictor and target indicators
- Document coverage maps: which indicators have how many country-years filled — this drives feasibility cuts before any modeling
- Output: a `data/indicators/inventory.parquet` and a markdown coverage report

### Phase 2 — Raw + Bronze (3–4 days)

- `etl/collect.py`: paginated OData ingestion, one Parquet per indicator code, retries with backoff
- `etl/bronze.py`: stack indicator files into a tidy country × year × indicator panel; reconcile mixed schemas via `union_by_name`
- Country-code normalization (ISO3) — GHO uses ISO3 already, but handle regional aggregates (`SEAR`, `EUR`, etc.) explicitly

### Phase 3 — Silver feature store (3–4 days)

- DuckDB SQL views computing point-in-time-correct temporal windows per country: lifetime, last 5y, last 10y, last 20y
- Trend slope, rolling mean, rolling std, year-over-year change for each indicator
- Lagged features (1y, 3y, 5y lags)
- Export per-window Parquet files matching the F1 project's `fs_driver_lifeN.parquet` pattern

### Phase 4 — Gold ABTs (2–3 days)

- One ABT SQL file per target — `abt_outbreak_risk.sql`, `abt_resurgence.sql`, `abt_vaccination_milestone.sql`
- Each ABT joins relevant silver features with the labeled outcome and explicitly excludes leakage features (no future-period predictors, no target-derived features)
- Document leakage decisions in code comments — recruiters reading the repo will look for this

### Phase 5 — ML training (4–5 days)

- One model module per target, mirroring `champion_model.py` / `team_model.py` / `departure_model.py`
- Candidates: LogisticRegression, BalancedRandomForest, LightGBM, XGBoost
- Optuna with TPE sampler and median pruner, all runs logged to MLFlow
- Time-based train/test split — never random, always by year (e.g. train ≤ 2018, test 2019–2023)

### Phase 6 — Streamlit app (3–4 days)

- Predictions tab: select target, country, horizon → show probability and historical trajectory
- Model comparison tab: same as F1 project (ROC, PR, confusion matrices, per-model metrics table)
- EDA tab: indicator explorer with Plotly
- **World map tab (new):** Folium choropleth of latest predicted probability per country; pydeck time-slider animation of historical incidence
- DuckDB console tab: SQL playground with 10–15 example queries

### Phase 7 — Polish + README + Docker (2 days)

- README with architecture diagram, screenshots, "what I learned" section
- Docker compose validates a clean spin-up
- Pin dependencies, write `.env.example`

### Phase 8 (v2) — Capacity strain target + forecasting

- Add Target 4 (healthcare system strain)
- Optionally add TimesFM zero-shot baseline as in F1 project, for direct comparison against trained models on the resurgence target

**Total v1 estimate:** ~3–4 weeks at a steady evening/weekend pace.

## Key Design Decisions (and the rationale to defend them in interviews)

1. **Annual cadence, not monthly.** GHO is annual. Forcing monthly granularity would mean fabricating data via interpolation, which corrupts the target. Honesty about cadence is a feature, not a bug.
2. **Country–year as the unit of observation.** Mirrors how WHO publishes and how policy operates.
3. **Time-based splits only.** Random splits leak future information across years and are the #1 mistake in time-aware ML projects.
4. **Leakage column audits.** Each ABT explicitly lists excluded features in code comments. The F1 project does this; VigiHealth must too.
5. **Multi-target, single lake.** One ETL pipeline serves all targets. This is the strongest part of the F1 project's design and worth preserving.
6. **Geographic viz earns its place.** A world map for health data isn't decoration — it surfaces sparseness and inequality patterns that tables hide.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Indicator coverage is sparser than expected for some countries | High | Phase 1 explicit coverage audit; drop low-coverage countries from training |
| Class imbalance is severe (rare outbreaks, rare resurgences) | High | imbalanced-learn (already in stack), PR-AUC as primary metric, calibration plots |
| GHO API rate-limits or goes down during ingestion | Medium | Caching to disk, idempotent ingestion, exponential backoff |
| Target 1 ("outbreak risk") feels weak after reframing to annual | Medium | Document the reframing clearly; offer Disease Outbreak News scraping as a v2 add-on |
| Scope creep to all 4 targets in v1 | Medium | This plan defers Target 4 to v2 — hold the line |

## Definition of Done (v1)

- [ ] All four medallion layers materialized end-to-end via `python -m etl.run_pipeline`
- [ ] Three ML targets trained, with at least 4 model candidates each, all logged to MLFlow
- [ ] Streamlit app runs locally and in Docker, with all five tabs functional
- [ ] World map tab renders predictions for the latest available year for at least one target
- [ ] README explains architecture, how to run, and key decisions
- [ ] Repo passes a clean `git clone && docker-compose up` test on a fresh machine
