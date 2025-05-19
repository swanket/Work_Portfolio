#%%
import os
import pandas as pd
import psycopg2

# --- Config ----------------------------------------------------------------
CSV_FOLDER = '/Users/sean/Work_Portfolio/Personal_Projects_and_Practice/PostGIS/CRD Weather Data GUI/stations'  # directory with station CSVs
DB_CONFIG = {
    'dbname': 'CRDclimate',
    'user': 'postgres',
    # 'password': '',
    'host': 'localhost',
    'port': 5432
}

# # --- Normalize headers ----------------------------------------------------
# rename_map = {
#     'air temp': 'air_temperature',
#     'air temperature': 'air_temperature',
#     'wind speed': 'wind_speed',
#     'wind dir': 'wind_direction',
#     'humidity (%)': 'relative_humidity',
#     'humidity': 'relative_humidity',
#     'swe': 'snow_water_equivalent',
#     'solar rad': 'solar_radiation',
#     'rainfall': 'rain',
#     'precip': 'precipitation',
# }

# --- Connect to DB --------------------------------------------------------
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()
#%%
# Fetch variable_id mapping
cursor.execute("SELECT name, variable_id FROM variables;")
var_map = dict(cursor.fetchall())

# --- Process each CSV -----------------------------------------------------
for filename in os.listdir(CSV_FOLDER):
    if not filename.endswith('.csv'):
        continue

    station_id = os.path.splitext(filename)[0]  # station from filename
    file_path = os.path.join(CSV_FOLDER, filename)

    print(f"Processing {filename} for station {station_id}...")

    # Load CSV
    df = pd.read_csv(file_path)

    # Clean & normalize column names
    # df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    # df.rename(columns=rename_map, inplace=True)

    if 'record_date' not in df.columns:
        print(f"Skipping {filename}: missing 'record_date'")
        continue

    df['station_id'] = station_id

    # Parse dates
    df['record_date'] = pd.to_datetime(df['record_date'])

    # Melt to long format
    melt = df.melt(
        id_vars=['station_id', 'record_date'],
        var_name='variable_name',
        value_name='value'
    )

    # Map variable_name to variable_id
    melt['variable_id'] = melt['variable_name'].map(var_map)

    # Drop unknown variables
    rows_before = len(melt)
    melt = melt.dropna(subset=['variable_id'])
    rows_after = len(melt)
    if rows_before != rows_after:
        print(f"  Dropped {rows_before - rows_after} rows with unknown variables.")

    # Insert rows
    insert_sql = """
        INSERT INTO readings (station_id, record_ts, variable_id, value)
        VALUES (%s, %s, %s, %s)
    """
    for _, row in melt.iterrows():
        cursor.execute(
            insert_sql,
            (
                row['station_id'],
                row['record_date'].to_pydatetime(),
                int(row['variable_id']),
                row['value']
            )
        )

    conn.commit()
    print(f"  Inserted {len(melt)} rows.")

# --- Cleanup --------------------------------------------------------------
cursor.close()
conn.close()
print("✅ All CSVs processed.")
