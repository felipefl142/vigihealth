-- Silver pivot: bronze long → country×year wide.
-- Stratification policy:
--   * dim1 IS NULL              — indicator has no strata, keep row
--   * dim1 = 'SEX_BTSX'         — both-sex aggregate (drop FMLE/MLE)
--   * dim1 = 'RESIDENCEAREATYPE_TOTL' — total population (drop URB/RUR splits)
-- Any remaining duplicates per (country, year, indicator) are averaged.
WITH filtered AS (
    SELECT country_iso3, year, indicator_code, value
    FROM panel
    WHERE dim1 IS NULL
       OR dim1 = 'SEX_BTSX'
       OR dim1 = 'RESIDENCEAREATYPE_TOTL'
),
deduped AS (
    SELECT country_iso3, year, indicator_code, AVG(value) AS value
    FROM filtered
    GROUP BY country_iso3, year, indicator_code
)
PIVOT deduped
ON indicator_code
USING FIRST(value)
GROUP BY country_iso3, year
ORDER BY country_iso3, year;
