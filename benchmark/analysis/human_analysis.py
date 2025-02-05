import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# External module imports
from embeding_visualization import get_embeddings
from benchmark.analysis.model_strategy import calculate_euclidean_distances


def load_data(database_name: str, base_path: str = "../user_database_sync/databases/"):
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


def process_games_df(games_df: pd.DataFrame):
    """
    Process the games DataFrame by adding calculated columns.
    """
    # Drop rows where player1Id or player2Id is None
    games_df = games_df.dropna(subset=['player1Id', 'player2Id'])

    # Create a new column 'Model Pair' treating player pairs symmetrically
    games_df['Model Pair'] = games_df.apply(
        lambda row: tuple(sorted([row['player1Id'], row['player2Id']])),
        axis=1
    )

    # Compute win flag for all games
    games_df['Win'] = games_df['status'].apply(lambda x: x == 'won')

    # Compute success rate grouped by model pair
    success_rate = games_df.groupby('Model Pair').agg(Success_Rate=('Win', 'mean'))

    # Filter only winning games and calculate round lengths
    wins_df = games_df[games_df['status'] == 'won'].copy()  # use copy to avoid warnings
    wins_df['Round Length 1'] = wins_df['wordsPlayed1'].apply(len)
    wins_df['Round Length 2'] = wins_df['wordsPlayed2'].apply(len)
    wins_df['Average Round Length'] = (wins_df['Round Length 1'] + wins_df['Round Length 2']) / 2

    # Compute average rounds for winning games grouped by model pair
    avg_rounds = wins_df.groupby('Model Pair').agg(Average_Rounds=('Average Round Length', 'mean'))

    # Merge the success rate and average rounds results
    result = success_rate.merge(avg_rounds, on='Model Pair')
    return result


def embedding_distance_analysis(games_df: pd.DataFrame, rounds: int = 6):
    """
    For winning games, compute the Euclidean distances between embeddings of the last few rounds.
    Then plot the mean and standard deviation of these distances over the rounds.
    """
    model_combinations = {}

    for _, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Embedding Analysis"):
        # Get embeddings for the last 'rounds' words played by both players
        embeddings_1 = get_embeddings(row['wordsPlayed1'][-rounds:])
        embeddings_2 = get_embeddings(row['wordsPlayed2'][-rounds:])

        if len(embeddings_1) >= rounds and len(embeddings_2) >= rounds:
            row_distances = [euclidean(embeddings_1[i], embeddings_2[i]) for i in range(rounds)]
            model_key = (row['player1Id'], row['player2Id'])
            model_combinations.setdefault(model_key, []).append(row_distances)

    # Plotting the results
    plt.figure(figsize=(14, 8))
    colors = plt.cm.jet(np.linspace(0, 1, len(model_combinations)))

    for color_index, (model_key, distances_lists) in enumerate(model_combinations.items()):
        if len(distances_lists) >= rounds:
            last_games = np.array(distances_lists[-rounds:])
            mean_of_last_games = np.mean(last_games, axis=0)
            std_of_last_games = np.std(last_games, axis=0)
            time_index = np.arange(rounds)

            plt.plot(
                time_index, mean_of_last_games,
                label=f'{model_key} Last {rounds}', marker='o',
                color=colors[color_index]
            )
            plt.fill_between(
                time_index,
                mean_of_last_games - std_of_last_games,
                mean_of_last_games + std_of_last_games,
                color=colors[color_index],
                alpha=0.3

            )

    plt.xlabel('Game Index')
    plt.ylabel('Average Euclidean Distance')
    plt.legend()
    plt.grid(True)
    plt.show()


def safe_calculate_euclidean_distances(row):
    distances = calculate_euclidean_distances(row)
    # Ensure the result is a list or tuple
    if not isinstance(distances, (list, tuple)):
        distances = [distances]
    # If fewer than 2 elements, pad with NaN; if more than 2, trim the list.
    if len(distances) < 2:
        distances = list(distances) + [np.nan] * (2 - len(distances))
    elif len(distances) > 2:
        distances = list(distances)[:2]
    return pd.Series(distances)


def strategy_analysis(games_df: pd.DataFrame, players_df: pd.DataFrame):
    """
    Analyze game strategy by computing the Euclidean distances using
    the 'calculate_euclidean_distances' function from the benchmark module.
    For each model, determine whether a mirroring or balancing strategy is predominant.
    """
    players = players_df['playerId'].unique()
    results = []

    tqdm.pandas(desc="Processing models")

    for player in players:
        # Select games (won or lost) involving the current model
        players_games = games_df[
            ((games_df['player1Id'] == player) | (games_df['player2Id'] == player))
        ].copy()

        # Calculate distances and expand into separate columns using progress_apply
        distances_series = players_games.progress_apply(safe_calculate_euclidean_distances, axis=1)
        distances_df = pd.DataFrame(distances_series, index=distances_series.index)
        players_games[['Distances to Previous', 'Distances to Average']] = distances_df

        # Compute average and standard deviation for both distance types
        players_games['Average Distance to Previous'] = players_games['Distances to Previous'].apply(
            lambda x: np.mean(x) if hasattr(x, 'size') and x.size else 0)
        players_games['Average Distance to Average'] = players_games['Distances to Average'].apply(
            lambda x: np.mean(x) if hasattr(x, 'size') and x.size else 0)
        players_games['Std Dev Distance to Previous'] = players_games['Distances to Previous'].apply(
            lambda x: np.std(x) if hasattr(x, 'size') and x.size else 0)
        players_games['Std Dev Distance to Average'] = players_games['Distances to Average'].apply(
            lambda x: np.std(x) if hasattr(x, 'size') and x.size else 0)

        mean_distance_to_previous = players_games['Average Distance to Previous'].mean()
        mean_distance_to_average = players_games['Average Distance to Average'].mean()
        std_distance_to_previous = players_games['Std Dev Distance to Previous'].mean()
        std_distance_to_average = players_games['Std Dev Distance to Average'].mean()
        sample_size = len(players_games)

        # Decide on strategy based on the comparison of the distances
        strategy = ("Mirroring Strategy" if mean_distance_to_previous < mean_distance_to_average
                    else "Balancing Strategy")

        results.append({
            "Model": player,
            "Mean Distance to Previous": mean_distance_to_previous,
            "Mean Distance to Average": mean_distance_to_average,
            "Std Dev Distance to Previous": std_distance_to_previous,
            "Std Dev Distance to Average": std_distance_to_average,
            "Number of Samples": sample_size,
            "Predominant Strategy": strategy
        })

    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    db_name = "downloaded_word_sync_20250204_165640.db"
    players_df, games_df = load_data(db_name)

    # Process game data to compute success rates and average rounds for winning games
    result = process_games_df(games_df)
    print("Success Rate and Average Rounds for Winning Games:")
    print(result)

    # Perform embedding distance analysis and plot the results
    embedding_distance_analysis(games_df, rounds=6)

    # Analyze strategies based on Euclidean distance calculations
    strategy_analysis(games_df, players_df)