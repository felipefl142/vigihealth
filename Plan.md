# VigiHealth — Project Plan (revised)

Multi-target global health analytics platform built on WHO Global Health Observatory (GHO) data. Mirrors the architecture of `f1-predict-analysis-platform`: a local-first medallion lake (DuckDB + Parquet) feeding ML models tracked in MLflow, served through a Streamlit web app — extended with country-level geographic visualization.

## Status (2026-05-02)

Scaffold complete: package layout, `constants.py`, stub modules in `etl/`, `ml/`, `app/`, `requirements.txt`, contract tests in `tests/`. No business logic yet.

This revision narrows v1 to **one target end-to-end** before broadening.

## Why VigiHealth

The name (Vigil + Health) signals continuous surveillance: detect, monitor, anticipate. Demonstrates the F1 project's data engineering and ML methodology applied to a domain with social impact and very different data characteristics — annual cadence, country-level entities, sparse and uneven indicator coverage.

## Architecture

```
WHO GHO OData API
      │
      ▼
data/raw/        Parquet per indicator (one file per IndicatorCode)
      │  DuckDB
      ▼
data/bronze/     Cleaned indicator panel (country × year × indicator, long/tidy)
      │
      ▼
data/silver/     Per-country temporal feature store (life, last 5y, last 10y, last 20y)
      │
      ▼
data/gold/       Analytical Base Tables (one per ML target)
      │
      ▼
ML Models  ──►  MLflow  ──►  Streamlit App (with world choropleth)
```

Differences from the F1 project:

| F1 project | VigiHealth |
|---|---|
| FastF1 (sessions, weather) | WHO GHO OData (~2300 indicators) |
| Race-level granularity | Country–year granularity |
| ~24 events × ~20 drivers per year | ~245 countries × ~25 years |
| In-season vs end-of-year ABTs | Forecast horizon variants (1y, 3y, 5y) |
| 3 targets shipped together | 1 target v1, then 2 more |

## Data Source — WHO GHO

**Endpoint:** `https://ghoapi.azureedge.net/api/`

- ~2300 indicators across 245 countries/regions, decades of history
- Mostly **annual** cadence — *not monthly*. Constrains target framing.
- No authentication, no documented rate limits (be polite).
- OData filtering: `$filter=SpatialDim eq 'BRA'`, `TimeDimensionBegin ge 2000`, etc.
- API does **not** provide COVID-19 data.
- Indicator metadata: `/api/Indicator`. Dimension values: `/api/DIMENSION/{name}/DimensionValues`.

**Data scale.** ~245 countries × ~25 years × ~50 indicators ≈ 300k rows in bronze. DuckDB is more than enough; chosen for SQL ergonomics and parity with the F1 project, not for scale.

## ML Targets

### v1 (MVP) — Target 3, Vaccination Coverage Milestone

*Will country C reach ≥90% coverage for vaccine V within horizon H years?*

- **Vaccines:** DTP3, MCV1 (measles), HepB3, Pol3
- **Horizons:** 1y, 3y, 5y (three label variants, same feature matrix)
- **Label:** binary — milestone achieved within H years from prediction year
- **Granularity:** country–year
- **Why this target first:** densest GHO coverage in the WHS4_* series, cleanest binary outcome, well-defined horizon, closest analog to F1's "champion" target.

**Pinned indicator codes:**

| Role | Code | Description |
|---|---|---|
| Target / feature | `WHS4_100` | DTP3 immunization coverage among 1-year-olds (%) |
| Target / feature | `WHS4_544` | MCV1 (measles) immunization coverage among 1-year-olds (%) |
| Target / feature | `WHS4_117` | HepB3 immunization coverage among 1-year-olds (%) |
| Target / feature | `WHS4_543` | Pol3 (polio) immunization coverage among 1-year-olds (%) |
| Covariate | `GHED_CHEGDP_SHA2011` | Current health expenditure as % of GDP |
| Covariate | `WHOSIS_000001` | Life expectancy at birth (years) |
| Covariate | `MDG_0000000007` | Under-5 mortality rate (per 1000 live births) |
| Covariate | `WSH_WATER_SAFELY_MANAGED` | Population using safely managed drinking-water services (%) |
| Covariate | `WSH_SANITATION_SAFELY_MANAGED` | Population using safely managed sanitation services (%) |

If any pinned code returns no data on `GET /api/{code}`, swap to the closest equivalent during Phase 1 scoping and update this table.

**Leakage rules:**
- For target `(vaccine V, horizon H)`, drop V's own coverage values within the horizon window.
- Do not include any feature derived from years > prediction year.
- Document each excluded column in `etl/sql/abt_vaccination_milestone.sql` with an inline comment.

### v1.1 — Target 2, Disease Resurgence Detection

*Will a disease that has been declining in country C for ≥5 years reverse trend in the next 1–3 years?*

- **Diseases:** measles, pertussis, polio (cases reported, not coverage)
- **Label:** binary — rolling 3-year mean of incidence rises above prior 5-year mean
- **Pinned indicator codes (provisional, confirm in Phase 1):**

| Role | Code | Description |
|---|---|---|
| Target | `WHS3_62` | Measles — number of reported cases |
| Target | `WHS3_41` | Pertussis — number of reported cases |
| Target | `WHS3_49` | Polio — number of reported cases |
| Feature | `WHS4_544`, `WHS4_543` | Vaccination coverage trajectories |

Build after v1 ships. Not before.

### v1.2 — Target 1, Outbreak Risk Forecast

*Will country C experience an above-trend incidence year for disease D in the next reporting year?*

- **Diseases:** tuberculosis, malaria
- **Label:** binary — incidence rate exceeds country's rolling 5-year median by >1 SD
- **Pinned indicator codes (provisional, confirm in Phase 1):**

| Role | Code | Description |
|---|---|---|
| Target | `MDG_0000000020` | Tuberculosis incidence (per 100k) |
| Target | `MALARIA_EST_INCIDENCE` | Malaria estimated incidence (per 1000 at risk) |
| Feature | covariates from v1 set | shared with vaccination model |

Build after v1.1.

### v2 (deferred) — Target 4, Healthcare System Strain

*Will country C's hospital bed density fall below a disease-burden-adjusted threshold within the next 5 years?*

Indicator: hospital bed density per 10k population (annual, structural). Sparser than the others. Threshold derivation needs methodology work. Defer until v1.0–v1.2 ship.

## Tech Stack

| Layer | Tool |
|---|---|
| Data collection | `requests` against the GHO OData endpoint |
| Storage | Local Parquet files (pyarrow) |
| SQL engine | DuckDB |
| Data processing | pandas |
| ML models | scikit-learn, XGBoost, LightGBM |
| Class balancing | imbalanced-learn |
| Hyperparameter tuning | Optuna (TPE sampler, median pruner) |
| Experiment tracking | MLflow with SQLite backend |
| Web app | Streamlit |
| Charts | Plotly, Matplotlib |
| Maps | Folium choropleth |
| Tests | pytest (contract tests + a few unit tests per layer) |

**Dropped from earlier draft:** Prophet, `ghoclient`, pydeck animations, Docker. Explicitly out of v1 scope.

**Kept as v2 backlog:** TimesFM zero-shot baseline (separate venv) for resurgence-target comparison.

## Project Structure

```
vigihealth/
├── app/
│   ├── main.py                       # Layout-only wiring
│   ├── tab_predictions.py
│   ├── tab_model_comparison.py
│   ├── tab_eda.py
│   ├── tab_world_map.py              # Folium choropleth
│   ├── tab_duckdb.py                 # SQL console
│   └── helpers.py
├── etl/
│   ├── collect.py                    # GHO OData ingestion
│   ├── bronze.py
│   ├── silver.py
│   ├── gold.py
│   ├── run_pipeline.py
│   └── sql/
│       ├── bronze_panel.sql
│       ├── fs_country_life.sql
│       ├── fs_country_last5.sql
│       ├── fs_country_last10.sql
│       ├── fs_country_last20.sql
│       ├── fs_country_all.sql
│       ├── abt_vaccination_milestone.sql   # v1
│       ├── abt_resurgence.sql              # v1.1
│       └── abt_outbreak_risk.sql           # v1.2
├── ml/
│   ├── model_selection.py
│   ├── vaccination_model.py          # v1
│   ├── resurgence_model.py           # v1.1
│   ├── outbreak_model.py             # v1.2
│   ├── predict.py
│   └── utils.py
├── tests/                            # contract + thin unit tests
├── data/                             # raw / bronze / silver / gold (gitignored)
├── mlruns/                           # gitignored
├── notebooks/                        # exploration only
├── constants.py
├── requirements.txt
├── README.md
├── Plan.md
├── DESIGN.md
└── CLAUDE.md
```

## Phased Roadmap

Phase 0 already complete (scaffold + tests). Picking up at Phase 1.

### Phase 1 — Indicator scoping (1–2 days)

Load-bearing. Get this wrong and the next four phases waste effort.

- Hit `/api/Indicator`, write to `data/indicators/inventory.parquet`
- For each pinned code in the v1 indicator table above, fetch a sample and confirm: code exists, returns rows, has expected dimensions
- Build a coverage matrix: country × year × indicator filled-rate. Drop countries below a coverage threshold from the training set (recorded in a config, not silently).
- Output: `notebooks/01_indicator_scoping.ipynb` with the coverage report.

### Phase 2 — Raw + Bronze (2–3 days)

- `etl/collect.py`: paginated OData ingestion, one Parquet per indicator code, retries with exponential backoff, idempotent (skip if file exists unless `--force`).
- `etl/bronze.py`: stack indicator files into a tidy `country_iso3 × year × indicator_code × value` panel via DuckDB `union_by_name`.
- Filter regional aggregates (`SEAR`, `EUR`, `AFR`, …) out of the modeling panel; keep them in a separate `bronze_regions.parquet` for the EDA tab.

### Phase 3 — Silver feature store (2–3 days)

- DuckDB SQL views computing point-in-time-correct temporal windows per country: lifetime, last 5y, last 10y, last 20y.
- Per-indicator features: lag(1, 3, 5), rolling mean, rolling std, trend slope, year-over-year change.
- Export per-window Parquet files; one wide row per (country, year).

### Phase 4 — Gold ABT for v1 target (1–2 days)

- `etl/sql/abt_vaccination_milestone.sql`: join silver features with vaccination milestone label for each `(vaccine, horizon)` pair.
- Explicit leakage exclusions in inline SQL comments.
- Output: `data/gold/abt_vaccination_milestone.parquet`.

### Phase 5 — v1 ML training (2–3 days)

- `ml/vaccination_model.py`: train on `abt_vaccination_milestone`.
- Candidates: LogisticRegression, BalancedRandomForest, LightGBM, XGBoost. `--nologreg` flag to skip LR for fast runs.
- Optuna with TPE sampler and median pruner. Default 50 trials dev, 200 final.
- Time-based split: train ≤ 2018, test ≥ 2019. Split year logged to MLflow.
- Primary metric PR-AUC. Also log ROC-AUC, F1, Brier, per-class confusion matrix, feature importance plot, holdout predictions CSV.

### Phase 6 — Streamlit v1 (2–3 days)

Five tabs, vaccination-only first. Each tab is its own module with a single `render()` function; `main.py` is layout-only.

- **Predictions:** select vaccine + country + horizon → probability + historical trajectory (Plotly)
- **Model Comparison:** ROC, PR, confusion matrices, per-model metrics table sourced from MLflow
- **EDA:** indicator explorer (country selector → time series)
- **World Map:** Folium choropleth of latest predicted milestone probability per country, with a year slider
- **DuckDB Console:** SQL playground with 8–10 example queries

`@st.cache_data` on every Parquet read.

### Phase 7 — README, screenshots, polish (1 day)

- Architecture diagram, screenshots, "what I learned" section
- Pin dependencies in `requirements.txt`
- `.env.example` with every env var used in code

**v1 estimate:** ~2 weeks at evening/weekend pace.

### Phase 8 — v1.1 (resurgence) and v1.2 (outbreak)

- Add ABTs `abt_resurgence.sql` and `abt_outbreak_risk.sql`
- Add training modules
- Extend the Predictions and World Map tabs to surface the additional targets

### Phase 9 (v2) — TimesFM baseline + Target 4

- TimesFM zero-shot forecast on resurgence target, in a separate venv (heavyweight install)
- Healthcare strain target with derived burden-adjusted threshold

## Testing Strategy

The scaffold already includes contract tests (`tests/test_repo_contract.py`, `tests/test_api_contract.py`). Layered approach going forward:

- **Contract tests** — module importability, required functions exist, constants resolve (already in place; keep green).
- **Unit tests** — per ETL layer: small synthetic frames in / expected frame out. Don't mock the DuckDB engine; use it on in-memory data.
- **Integration smoke** — one test per phase output: `data/bronze/panel.parquet` exists and matches expected schema after `python -m etl.bronze` on a 5-indicator subset. Skipped in CI if data missing.
- **No mock-only ML tests.** Train on a tiny fixture ABT and assert that PR-AUC > random baseline.

## Key Design Decisions

1. **Annual cadence, not monthly.** GHO is annual. Forcing monthly via interpolation fabricates data.
2. **Country–year as the unit of observation.** Mirrors how WHO publishes and how policy operates.
3. **Time-based splits only.** Random splits leak future across years.
4. **Leakage column audits in SQL comments.** Reviewers look for this.
5. **One target end-to-end before fanning out.** Prove the loop on vaccination, then replicate for resurgence and outbreak.
6. **Multi-target, single lake.** One ETL pipeline serves all targets.
7. **Geographic viz limited to Folium choropleth in v1.** pydeck animations deferred — extra dependency, marginal v1 value.
8. **No Docker in v1.** No deploy target yet; local-first solo dev. Revisit when packaging for reviewers or deploying to Streamlit Cloud / HF Spaces.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Indicator coverage sparser than expected | High | Phase 1 coverage audit; drop low-coverage countries from training set, recorded in config |
| Class imbalance severe (rare milestones in low-coverage countries) | High | imbalanced-learn, PR-AUC primary, calibration plots |
| GHO API down or slow during ingestion | Medium | Idempotent ingestion, exponential backoff, skip-if-exists |
| One of the pinned indicator codes is wrong | Medium | Phase 1 sample fetch validates each code; swap and update Plan if needed |
| Scope creep back to "all targets at once" | Medium | This revision exists to prevent that. Hold the v1 line. |

## Definition of Done — v1

- [ ] All four medallion layers materialized end-to-end via `python -m etl.run_pipeline`
- [ ] Vaccination milestone target trained with ≥4 model candidates, all logged to MLflow
- [ ] Streamlit app runs locally with all five tabs functional
- [ ] World Map tab renders predicted milestone probability per country for the latest horizon
- [ ] README explains architecture, how to run, and key decisions
- [ ] `pytest` green
