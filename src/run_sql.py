"""Run every query in sql/queries.sql against the SQLite database and print the results."""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.db"
SQL_PATH = PROJECT_ROOT / "sql" / "queries.sql"


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("Database not found. Run: python src/load_to_sqlite.py")

    statements = SQL_PATH.read_text().split(";")

    with sqlite3.connect(DB_PATH) as conn:
        for stmt in statements:
            stmt = stmt.strip()
            # Drop comment lines to get the executable part
            code = "\n".join(
                line for line in stmt.splitlines() if not line.strip().startswith("--")
            ).strip()
            if not code:
                continue
            print(stmt, "\n")
            print(pd.read_sql_query(code, conn).to_string(index=False))
            print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()