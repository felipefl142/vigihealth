-- Bronze panel: tidy long format country × year × indicator.
-- One row per (country, year, indicator, dim1, dim2, dim3) tuple.
-- Source: every Parquet under data/raw/. Stacked via union_by_name to tolerate
-- mixed schemas across indicators (some carry Dim1 stratifications, others not).
-- Filters:
--   * SpatialDimType = 'COUNTRY' — regional/income-group/global aggregates land
--     in bronze_regions.parquet, not the modeling panel.
--   * NumericValue IS NOT NULL — string-only Value rows are not numeric and
--     cannot be aggregated.
-- Output columns are stable; indicator-specific dimensions stay in dim1/dim2/dim3.
SELECT
    SpatialDim                              AS country_iso3,
    CAST(TimeDim AS INTEGER)                AS year,
    IndicatorCode                           AS indicator_code,
    NumericValue                            AS value,
    Dim1                                    AS dim1,
    Dim2                                    AS dim2,
    Dim3                                    AS dim3,
    ParentLocationCode                      AS region_code,
FROM read_parquet(?, union_by_name=true)
WHERE SpatialDimType = 'COUNTRY'
  AND NumericValue IS NOT NULL
  AND TimeDim IS NOT NULL;
