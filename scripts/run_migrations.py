"""Execute database migrations in numerical order."""

import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings

load_dotenv(ROOT_DIR / ".env")


def get_connection():
    return psycopg2.connect(
        host=settings.get_db_host(),
        port=settings.get_db_port(),
        dbname=settings.get_db_name(),
        user=settings.get_db_user(),
        password=settings.get_db_password(),
    )


def run_migrations():
    migrations_dir = ROOT_DIR / "migrations"
    if not migrations_dir.exists():
        print("No migrations directory found.")
        return

    sql_files = sorted(f for f in migrations_dir.glob("*.sql") if re.match(r"^\d{3}_", f.name))

    if not sql_files:
        print("No migration files found.")
        return

    conn = get_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    for sql_file in sql_files:
        print(f"Running migration: {sql_file.name}")
        sql = sql_file.read_text(encoding="utf-8")

        try:
            cursor.execute(sql)
        except psycopg2.Error as err:
            print(f"  Error in {sql_file.name}: {err}")
            raise

        print(f"  \u2713 {sql_file.name} applied")

    cursor.close()
    conn.close()
    print("\nAll migrations applied successfully.")


if __name__ == "__main__":
    run_migrations()
