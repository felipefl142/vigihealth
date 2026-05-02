-- ABT: vaccination milestone.
-- Question: will country C reach >= MILESTONE_THRESHOLD coverage for vaccine V
-- within HORIZON years of prediction year Y?
--
-- Construction:
--   * Predictors    = all silver features at (country, year=Y).
--   * Future panel  = same wide pivot but restricted to the four vaccine codes,
--                     used only to derive the label.
--   * Label         = MAX(coverage_V over years Y+1 .. Y+H) >= threshold.
--   * Leakage rules:
--       - Silver features are point-in-time correct (year <= Y), so no future
--         feature can leak into row Y. Verified in etl/silver.py.
--       - The label is derived strictly from years Y+1..Y+H — never from year Y.
--       - Vaccine V's *own* current value (WHS4_*_value) is kept as a predictor
--         deliberately: knowing the most recent coverage is fair information.
--         All values within the horizon window are excluded by construction
--         since the future panel is a separate join keyed on Y+offset.
--   * Rows where the future window is partially or fully missing are dropped:
--     no label, no training row.
WITH future_panel AS (
    SELECT country_iso3, year,
           "WHS4_100" AS dtp3,
           "WHS4_544" AS mcv1,
           "WHS4_117" AS hepb3,
           "WHS4_543" AS pol3
    FROM read_parquet(?)
),
labels AS (
    SELECT
        country_iso3,
        year,
        -- horizon 1y: just next year
        MAX(dtp3)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING) AS dtp3_h1_max,
        MAX(mcv1)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING) AS mcv1_h1_max,
        MAX(hepb3) OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING) AS hepb3_h1_max,
        MAX(pol3)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING) AS pol3_h1_max,
        -- horizon 3y: any of next 3 years
        MAX(dtp3)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS dtp3_h3_max,
        MAX(mcv1)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS mcv1_h3_max,
        MAX(hepb3) OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS hepb3_h3_max,
        MAX(pol3)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS pol3_h3_max,
        -- horizon 5y: any of next 5 years
        MAX(dtp3)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) AS dtp3_h5_max,
        MAX(mcv1)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) AS mcv1_h5_max,
        MAX(hepb3) OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) AS hepb3_h5_max,
        MAX(pol3)  OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) AS pol3_h5_max
    FROM future_panel
)
SELECT * FROM labels;
