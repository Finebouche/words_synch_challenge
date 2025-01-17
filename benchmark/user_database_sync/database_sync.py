import os
import requests
from datetime import datetime

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

if __name__ == '__main__':
    token_path = 'download_db_key.txt'  # Path to your local token file
    download_url = 'https://word-sync.games//database/download-database'  # URL to download the database

    # Name of the subdirectory for storing downloaded databases
    subdir = 'databases'

    # Read the token from the file
    try:
        with open(token_path, 'r') as file:
            token = file.read().strip()
    except FileNotFoundError:
        print("Token file not found.")
        exit()

    # Download the database into the subdirectory
    download_database(download_url, token, subdir)