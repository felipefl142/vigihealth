-- Bronze regions: regional, income-group, and global aggregates.
-- Excluded from modeling (would leak country-level signal); kept for the EDA tab.
SELECT
    SpatialDim                              AS region_code,
    SpatialDimType                          AS region_type,
    CAST(TimeDim AS INTEGER)                AS year,
    IndicatorCode                           AS indicator_code,
    NumericValue                            AS value,
    Dim1                                    AS dim1,
    Dim2                                    AS dim2,
    Dim3                                    AS dim3,
FROM read_parquet(?, union_by_name=true)
WHERE SpatialDimType <> 'COUNTRY'
  AND NumericValue IS NOT NULL
  AND TimeDim IS NOT NULL;
