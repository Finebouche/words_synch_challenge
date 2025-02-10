import os
import requests
from datetime import datetime
import sqlite3

def download_database(url, token, directory):
    """
    Download the SQLite database using an authentication token
    and save it to a subdirectory (with a timestamp).
    """
    # Ensure the subdirectory exists
    os.makedirs(directory, exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_filename = f"downloaded_word_sync_{timestamp}.db"
    db_path = os.path.join(directory, db_filename)

    headers = {'x-download-token': token}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        with open(db_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
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
    db1_path = os.path.join(subdir, database1)
    db2_path = os.path.join(subdir, database2)
    output_path = os.path.join(subdir, output_database)

    # Create or connect to the output database
    conn_out = sqlite3.connect(output_path)
    cursor_out = conn_out.cursor()

    # Attach both source databases
    cursor_out.execute(f"ATTACH DATABASE '{db1_path}' AS db1")
    cursor_out.execute(f"ATTACH DATABASE '{db2_path}' AS db2")

    # Function to handle table copying
    def copy_tables(db_prefix):
        cursor_tables = cursor_out.execute(f"SELECT name FROM {db_prefix}.sqlite_master WHERE type='table'")
        tables = cursor_tables.fetchall()

        for table in tables:
            table_name = table[0]

            # Check if table already exists in the output database
            if cursor_out.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()[0] == 0:
                schema_query = cursor_out.execute(
                    f"SELECT sql FROM {db_prefix}.sqlite_master WHERE type='table' AND name='{table_name}'"
                ).fetchone()
                if schema_query:
                    cursor_out.execute(schema_query[0])  # Create table in output DB

            # Insert data (avoid duplicates using INSERT OR IGNORE)
            cursor_out.execute(f"INSERT OR IGNORE INTO {table_name} SELECT * FROM {db_prefix}.{table_name}")

    # Copy tables from both databases
    copy_tables("db1")
    copy_tables("db2")

    # Commit changes and close connection
    conn_out.commit()
    conn_out.close()

    print(f"Databases {database1} and {database2} successfully merged into {output_database}")


if __name__ == '__main__':
    token_path = 'download_db_key.txt'  # Path to your local token file
    # Read the token from the file
    try:
        with open(token_path, 'r') as file:
            token = file.read().strip()
    except FileNotFoundError:
        print("Token file not found.")
        exit()

    subdir = 'databases'

    # merge two databases
    # database1 = "downloaded_word_sync_20250205_161200.db"
    # database2 = "downloaded_word_sync_20250206_144759.db"
    # output_database = "merged.db"
    # combine_databases(subdir, database1, database2, output_database)

    # # Download the database into the subdirectory
    download_url = 'https://word-sync.games//database/download-database'  # URL to download the database
    download_database(download_url, token, subdir)