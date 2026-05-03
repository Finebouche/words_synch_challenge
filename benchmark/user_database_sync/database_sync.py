from datetime import datetime
from pathlib import Path
import sqlite3

import requests


BASE_DIR = Path(__file__).resolve().parent


def quote_identifier(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def download_database(url, token, directory):
    """
    Download the SQLite database using an authentication token
    and save it to a subdirectory (with a timestamp).
    """
    # Ensure the subdirectory exists
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_filename = f"downloaded_word_sync_{timestamp}.db"
    db_path = directory / db_filename

    headers = {'x-download-token': token}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        with db_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Database downloaded successfully and saved to: {db_path}")
    else:
        print("Failed to download database:", response.status_code, response.text)
        return False

    return True


def combine_databases(subdir, database1, database2, output_database):
    """
    Combine two SQLite databases into a single database.
    """
    subdir = Path(subdir)
    db1_path = subdir / database1
    db2_path = subdir / database2
    output_path = subdir / output_database

    with sqlite3.connect(output_path) as conn_out:
        cursor_out = conn_out.cursor()

        cursor_out.execute("ATTACH DATABASE ? AS db1", (str(db1_path),))
        cursor_out.execute("ATTACH DATABASE ? AS db2", (str(db2_path),))

        def copy_tables(db_prefix):
            cursor_tables = cursor_out.execute(
                f"SELECT name FROM {quote_identifier(db_prefix)}.sqlite_master WHERE type = 'table'"
            )
            tables = cursor_tables.fetchall()

            for (table_name,) in tables:
                safe_table_name = quote_identifier(table_name)

                table_exists = cursor_out.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,),
                ).fetchone()[0]

                if table_exists == 0:
                    schema_query = cursor_out.execute(
                        f"SELECT sql FROM {quote_identifier(db_prefix)}.sqlite_master "
                        "WHERE type = 'table' AND name = ?",
                        (table_name,),
                    ).fetchone()
                    if schema_query:
                        cursor_out.execute(schema_query[0])

                cursor_out.execute(
                    f"INSERT OR IGNORE INTO {safe_table_name} "
                    f"SELECT * FROM {quote_identifier(db_prefix)}.{safe_table_name}"
                )

        copy_tables("db1")
        copy_tables("db2")

    print(f"Databases {database1} and {database2} successfully merged into {output_database}")


if __name__ == '__main__':
    token_path = BASE_DIR / 'download_db_key.txt'
    try:
        token = token_path.read_text().strip()
    except FileNotFoundError:
        print("Token file not found.")
        raise SystemExit(1)

    subdir = BASE_DIR / 'databases'

    # merge two databases
    # database1 = "downloaded_word_sync_20250205_161200.db"
    # database2 = "downloaded_word_sync_20250206_144759.db"
    # output_database = "merged.db"
    # combine_databases(subdir, database1, database2, output_database)

    # # Download the database into the subdirectory
    download_url = 'https://word-sync.games/database/download-database'
    download_database(download_url, token, subdir)
