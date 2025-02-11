
from sqlalchemy import create_engine
import pandas as pd
import json


def load_sql_data(database_name: str, base_path: str = "../user_database_sync/databases/", player_ids=None):
    """
    Connect to the SQLite database and load Players and Games tables.
    Also converts the JSON strings for word lists back into Python lists.
    """
    DATABASE_PATH = base_path + database_name
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")

    players_df = pd.read_sql("SELECT * FROM Players", con=engine)

    if player_ids:
        formatted_ids = ', '.join([f"'{player_id}'" for player_id in player_ids])
        games_query = f"""
        SELECT * FROM Games
        WHERE player1Id IN ({formatted_ids}) OR player2Id IN ({formatted_ids})
        """
        games_df = pd.read_sql(games_query, con=engine)
    else:
        games_df = pd.read_sql("SELECT * FROM Games", con=engine)

    # Convert JSON stored as text back into lists for each game
    games_df['wordsPlayed1'] = games_df['wordsPlayed1'].apply(json.loads)
    games_df['wordsPlayed2'] = games_df['wordsPlayed2'].apply(json.loads)

    # Convert surveyAnswers columns from JSON strings to Python objects
    games_df['surveyAnswers1'] = games_df['surveyAnswers1'].apply(
        lambda x: json.loads(x) if pd.notnull(x) and x != "" else [])
    games_df['surveyAnswers2'] = games_df['surveyAnswers2'].apply(
        lambda x: json.loads(x) if pd.notnull(x) and x != "" else [])

    return players_df, games_df


def load_csv(path: str):
    """
    Load a CSV file from the given path.
    """
    games_df = pd.read_csv(path)

    return games_df

if __name__ == '__main__':
    import gensim.downloader as api

    print(api.base_dir)