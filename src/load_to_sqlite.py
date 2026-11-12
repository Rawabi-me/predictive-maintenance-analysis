"""Load the AI4I 2020 CSV into a SQLite database (table: machine_data)."""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
DB_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.db"

COLUMN_NAMES = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {CSV_PATH}. See data/README.md for download instructions."
        )

    df = pd.read_csv(CSV_PATH).rename(columns=COLUMN_NAMES)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("machine_data", conn, if_exists="replace", index=False)
        n_rows, n_failures = conn.execute(
            "SELECT COUNT(*), SUM(machine_failure) FROM machine_data"
        ).fetchone()

    print(f"Loaded {n_rows} rows into {DB_PATH.name} (table: machine_data)")
    print(f"Failures in table: {n_failures}")


if __name__ == "__main__":
    main()