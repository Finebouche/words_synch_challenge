import requests
import sqlite3
from sqlite3 import Error


def download_database(url, token, db_path):
    """Download the SQLite database using an authentication token."""
    headers = {'x-download-token': token}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        with open(db_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Database downloaded successfully.")
    else:
        print("Failed to download database:", response.status_code, response.text)
        return False
    return True


def query_database(db_path):
    """Query the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print("Connected to the database.")
        cur = conn.cursor()

        # Example query
        cur.execute("SELECT * FROM tablename")  # Replace 'tablename' with your actual table name
        rows = cur.fetchall()

        for row in rows:
            print(row)

        conn.close()
    except Error as e:
        print("Error during connection or query execution:", e)



if __name__ == '__main__':
    token_path = 'download_db_key.txt'  # Path to your local token file
    database_path = 'downloaded_word_sync.db'  # Local path to save the downloaded database
    download_url = 'https://word-sync.games//database/download-database'  # URL to download the database

    # Read the token from the file
    try:
        with open(token_path, 'r') as file:
            token = file.read().strip()
    except FileNotFoundError:
        print("Token file not found.")
        exit()

    # Download the database
    if download_database(download_url, token, database_path):
        # Query the database
        query_database(database_path)
