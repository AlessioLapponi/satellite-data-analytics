import sqlite3
from pathlib import Path

import pandas as pd

from typing import Union

def init_db(
    db_path: Union[str, Path] = "data/processed/satellites.db",
    csv_path: Union[str, Path] = "data/raw/satellite_orbital_catalog.csv",
):
    # 1) Paths
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "raw" / "satellite_orbital_catalog.csv"
    db_path = project_root / "data" / "processed" / "satellites.db"
    sql_schema_path = project_root / "sql" / "create_tables.sql"

    print("CSV path:", csv_path)
    print("DB path:", db_path)

    # 2) Read CSV
    df = pd.read_csv(csv_path)

    print("Columns in CSV:", df.columns.tolist())
    print("Raws in CSV:", len(df))

    # 3) Create database from SQL scheme
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    with open(sql_schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    cur.executescript(schema_sql)
    conn.commit()

    # 4) Prepare data
    
    expected_columns = [
        "norad_id",
        "name",
        "object_type",
        "satellite_constellation",
        "altitude_km",
        "altitude_category",
        "orbital_band",
        "congestion_risk",
        "inclination",
        "eccentricity",
        "launch_year_estimate",
        "days_in_orbit_estimate",
        "orbit_lifetime_category",
        "mean_motion",
        "epoch",
        "data_source",
        "snapshot_date",
        "country",
        "last_seen",
    ]

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns defined on the scheme are missing: {missing_cols}")

    # 5) Insert placeholders
    placeholders = ", ".join(["?"] * len(expected_columns))
    cols_sql = ", ".join(expected_columns)
    insert_sql = f"INSERT INTO satellites ({cols_sql}) VALUES ({placeholders})"

    # 6) Convert dataframe in n-tuple list
    rows = df[expected_columns].itertuples(index=False, name=None)

    # 7) Insert records in a block
    cur.executemany(insert_sql, rows)
    conn.commit()
    conn.close()

    print("db created and populated with sql explicit scheme")

if __name__ == "__main__":
    init_db()

def load_table(db_path: Union[str, Path], table_name: str = "satellites_clean") -> pd.DataFrame:
    #load SQL table in pandas
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    finally:
        conn.close()
    return df