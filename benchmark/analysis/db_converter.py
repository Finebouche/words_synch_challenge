import pandas as pd
import json
from sqlalchemy import create_engine


def split_words_row(row):
    """
    Splits the wordsArray field into wordsPlayed1 and wordsPlayed2.

    For human-vs-human play (i.e. when wordsArray is a list of dictionaries),
    extract the values for 'player1' and 'player2' from each dictionary.

    For bot play (i.e. when wordsArray is a list of strings and botId is provided),
    assign the even-indexed words (starting at index 0) to wordsPlayed2 (first word is from player2)
    and the odd-indexed words to wordsPlayed1.

    If the wordsArray field is stored as a JSON string, it is decoded first.
    """
    data = row['wordsArray']
    bot_id = row.get('botId')

    # If data is a JSON string, decode it.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            print("Error decoding JSON in row {}: {}".format(row.get('gameId', 'unknown'), e))
            return pd.Series({'wordsPlayed1': [], 'wordsPlayed2': []})

    # If data isn't a list, return empty lists.
    if not isinstance(data, list):
        return pd.Series({'wordsPlayed1': [], 'wordsPlayed2': []})

    # Case 1: Human vs. Human play (list of dictionaries)
    if data and isinstance(data[0], dict):
        wordsPlayed1 = [entry.get("player1") for entry in data]
        wordsPlayed2 = [entry.get("player2") for entry in data]
        return pd.Series({'wordsPlayed1': wordsPlayed1, 'wordsPlayed2': wordsPlayed2})

    # Case 2: Bot play (list of strings)
    elif data and isinstance(data[0], str):
        if bot_id:
            # For bot play, assign alternating words:
            # Even-indexed elements (0, 2, 4, …) go to wordsPlayed2 (first word is from player2)
            # Odd-indexed elements (1, 3, 5, …) go to wordsPlayed1.
            wordsPlayed2 = data[0::2]
            wordsPlayed1 = data[1::2]
        else:
            # If no botId is provided, default to all words in wordsPlayed1.
            wordsPlayed1 = data
            wordsPlayed2 = []
        return pd.Series({'wordsPlayed1': wordsPlayed1, 'wordsPlayed2': wordsPlayed2})

    # Fallback: return empty lists.
    return pd.Series({'wordsPlayed1': [], 'wordsPlayed2': []})


def main():
    # Database configuration: update paths as needed.
    db_name = "downloaded_word_sync_20250204_165640.db"
    db_path = "../user_database_sync/databases/" + db_name

    # Create a SQLAlchemy engine for the SQLite database.
    engine = create_engine(f"sqlite:///{db_path}")

    # Load the Games table into a DataFrame.
    games_df = pd.read_sql("SELECT * FROM Games", con=engine)

    # Apply the splitting function row-wise to create the new columns.
    split_columns = games_df.apply(split_words_row, axis=1)
    games_df['wordsPlayed1'] = split_columns['wordsPlayed1']
    games_df['wordsPlayed2'] = split_columns['wordsPlayed2']

    # Convert the lists into JSON strings (SQLite doesn't support Python lists directly).
    games_df['wordsPlayed1'] = games_df['wordsPlayed1'].apply(json.dumps)
    games_df['wordsPlayed2'] = games_df['wordsPlayed2'].apply(json.dumps)

    # (Optional) Drop the original wordsArray column if no longer needed.
    # games_df = games_df.drop(columns=['wordsArray'])

    # Write the updated DataFrame to a new table in the database (here named "Games_new").
    games_df.to_sql("Games", con=engine, if_exists="replace", index=False)

    print("Conversion complete. New table 'Games_new' created with wordsPlayed1 and wordsPlayed2 columns.")


if __name__ == "__main__":
    main()