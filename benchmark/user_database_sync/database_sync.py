import sqlite3
from sqlite3 import Error

def read_token_from_file(file_path):
    """Reads the authentication token from a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Token file not found.")
        return None


def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("Connection established.")
    except Error as e:
        print(e)
    return conn


def execute_query(conn, token, correct_token):
    """Execute a query if token is valid."""
    if token != correct_token:
        print("Invalid token.")
        return

    try:
        cur = conn.cursor()
        # Example query: Select everything from a hypothetical 'users' table
        cur.execute("SELECT * FROM users")

        rows = cur.fetchall()
        for row in rows:
            print(row)
    except Error as e:
        print(e)



if __name__ == '__main__':
    database_path = 'word_sync.db'  # Path to your SQLite database
    token_file_path = 'download_db_key.txt'  # Path to the file containing the token

    # Read token from file
    token = read_token_from_file(token_file_path)
    if token is None:
        exit()

    # Create a database connection
    conn = create_connection(database_path)
    if conn:
        correct_token = 'your_secret_token'  # This should match the token in your token.txt
        execute_query(conn, token, correct_token)

        conn.close()  # Close the connection
