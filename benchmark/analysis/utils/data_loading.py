
from sqlalchemy import create_engine
import pandas as pd
import json


def load_sql_data(database_name: str, base_path: str = "../user_database_sync/databases/"):
    """
    Connect to the SQLite database and load Players and Games tables.
    Also converts the JSON strings for word lists back into Python lists.
    """
    DATABASE_PATH = base_path + database_name
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")

    players_df = pd.read_sql("SELECT * FROM Players", con=engine)
    games_df = pd.read_sql("SELECT * FROM Games", con=engine)

    # Convert JSON stored as text back into lists for each game
    games_df['wordsPlayed1'] = games_df['wordsPlayed1'].apply(json.loads)
    games_df['wordsPlayed2'] = games_df['wordsPlayed2'].apply(json.loads)

    return players_df, games_df

def load_csv(path: str):
    """
    Load a CSV file from the given path.
    """
    games_df = pd.read_csv(path)

    games_df['wordsPlayed1'] = games_df['wordsPlayed1'].apply(json.loads)
    games_df['wordsPlayed2'] = games_df['wordsPlayed2'].apply(json.loads)

    return games_df

if __name__ == '__main__':
    import gensim.downloader as api

    print(api.base_dir)