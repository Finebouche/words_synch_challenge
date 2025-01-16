import requests

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
    download_database(download_url, token, database_path)