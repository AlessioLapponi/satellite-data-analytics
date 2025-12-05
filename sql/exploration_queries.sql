-- Q1) Count each object type
SELECT object_type, COUNT(*) AS n_objects
FROM satellites_clean
GROUP BY object_type
ORDER BY n_objects DESC;

-- Q2) altitude bins by object type
SELECT
    object_type,
    CASE
        WHEN altitude_km < 1000 THEN '0–1000 km'
        WHEN altitude_km < 20000 THEN '1000–20000 km'
        WHEN altitude_km < 36000 THEN '20000–36000 km'
        ELSE '>= 36000 km'
    END AS altitude_bin,
    COUNT(*) AS n_objects,
    AVG(eccentricity) AS avg_eccentricity
FROM satellites_clean
WHERE altitude_km IS NOT NULL
GROUP BY object_type, altitude_bin
ORDER BY altitude_bin, object_type;

-- Q3) Distribution per orbital class
SELECT orbital_band, COUNT(*) AS n_objects
FROM satellites_clean
GROUP BY orbital_band
ORDER BY n_objects DESC;

-- Q4) Altitude and eccentricity per orbital class
SELECT orbital_band,
       COUNT(*)           AS n_objects,
       AVG(altitude_km)   AS avg_altitude_km,
       MIN(altitude_km)   AS min_altitude_km,
       MAX(altitude_km)   AS max_altitude_km,
       AVG(eccentricity)  AS avg_eccentricity
FROM satellites_clean
GROUP BY orbital_band
ORDER BY avg_altitude_km;

-- Q5) Countries with more payloads
SELECT country, COUNT(*) AS n_payloads
FROM satellites_clean
WHERE object_type = 'PAYLOAD'
GROUP BY country
HAVING country IS NOT NULL
ORDER BY n_payloads DESC
LIMIT 10;

-- Q6) Bucket of eccentricity and altitude
SELECT
    CASE
        WHEN eccentricity < 0.01 THEN 'quasi-circular'
        WHEN eccentricity < 0.1  THEN 'low-eccentricity'
        WHEN eccentricity < 0.5  THEN 'medium-eccentricity'
        ELSE 'high-eccentricity'
    END AS ecc_bin,
    COUNT(*)                AS n_objects,
    AVG(altitude_km)        AS avg_altitude_km,
    AVG(inclination)    AS avg_inclination_deg
FROM satellites_clean
WHERE eccentricity IS NOT NULL
GROUP BY ecc_bin
ORDER BY n_objects DESC;

-- Q7) Data for altitude vs eccentricity scatter
SELECT altitude_km, eccentricity, object_type
FROM satellites_clean
WHERE altitude_km IS NOT NULL
  AND eccentricity IS NOT NULL;