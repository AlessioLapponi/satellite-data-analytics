-- sql/create_tables.sql

DROP TABLE IF EXISTS satellites;

CREATE TABLE satellites (
    norad_id                INTEGER PRIMARY KEY,
    name                    TEXT,
    object_type             TEXT,      -- PAYLOAD / DEBRIS / ROCKET BODY
    satellite_constellation TEXT,
    altitude_km             REAL,
    altitude_category       TEXT,
    orbital_band            TEXT,
    congestion_risk         TEXT,
    inclination             REAL,
    eccentricity            REAL,
    launch_year_estimate    INTEGER,
    days_in_orbit_estimate  INTEGER,
    orbit_lifetime_category TEXT,
    mean_motion             REAL,
    epoch                   TEXT,
    data_source             TEXT,
    snapshot_date           TEXT,
    country                 TEXT,
    last_seen               TEXT
);