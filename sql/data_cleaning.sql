DROP TABLE IF EXISTS satellites_clean;

CREATE TABLE satellites_clean AS
SELECT
    *,
    -- Calculate orbital_period from mean_motion
    (1440.0 / mean_motion) AS period_minutes
FROM satellites
WHERE
    inclination IS NOT NULL
    AND eccentricity IS NOT NULL
    AND altitude_km IS NOT NULL
    AND mean_motion IS NOT NULL
    AND object_type IN ('PAYLOAD', 'DEBRIS', 'ROCKET BODY');