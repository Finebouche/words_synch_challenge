import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

# External module imports
from embeding_utils import get_openai_embeddings, get_embeddings, get_openai_embedding, load_model
from data_loading import load_sql_data

from scipy.spatial.distance import cosine

def calculate_player_metrics(games_df):
    # Filter games where status is 'won' to calculate success rates
    won_games = games_df[games_df['status'] == 'won']

    # Calculate human_success_rate
    human_games = won_games[won_games['botId'].isna()]  # Games without a bot involved
    human_success_count = human_games['player1Id'].value_counts().add(human_games['player2Id'].value_counts(), fill_value=0)
    total_human_games = games_df[games_df['botId'].isna()]['player1Id'].value_counts().add(games_df[games_df['botId'].isna()]['player2Id'].value_counts(), fill_value=0)
    human_success_rate = (human_success_count / total_human_games).fillna(0)

    # Calculate bot_success_rate
    bot_games = won_games[won_games['player2Id'].isna()]  # Assuming bots only play in player2Id's slot
    bot_success_count = bot_games['player1Id'].value_counts()
    total_bot_games = games_df[games_df['player2Id'].isna()]['player1Id'].value_counts()
    bot_success_rate = (bot_success_count / total_bot_games).fillna(0)

    # Calculate average number of rounds per player
    total_rounds = games_df.groupby('player1Id')['roundCount'].sum().add(games_df.groupby('player2Id')['roundCount'].sum(), fill_value=0)
    total_games_per_player = games_df['player1Id'].value_counts().add(games_df['player2Id'].value_counts(), fill_value=0)
    average_num_round = (total_rounds / total_games_per_player).fillna(0)

    # Combine all metrics into a single DataFrame
    metrics_df = pd.DataFrame({
        'Human Success Rate': human_success_rate,
        'Bot Success Rate': bot_success_rate,
        'Average Number of Rounds': average_num_round
    })

    return metrics_df

def get_embeddings_for_table(games_df: pd.DataFrame, model_name="openai"):
    """
    Get embeddings for the last words played by each player in each game.
    """
    # check that games_df doesn't contain the embeddings already

    if 'embedding1_' + model_name in games_df.columns and 'embedding2_' + model_name in games_df.columns:
        return games_df

    if model_name == "openai":
        embedding_model = None
    elif model_name == "word2vec" or model_name == "glove":
        embedding_model = load_model(model_name=model_name)
    else:
        raise ValueError("Unsupported model. Choose 'openai', 'word2vec' or 'glove'")

    # Iterate through each game
    embeddings = []
    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Fetching Embeddings"):
        words_player1 = row['wordsPlayed1']
        words_player2 = row['wordsPlayed2']

        # Ensure both players have played words
        if len(words_player1) > 0 and len(words_player2) > 0:
            # Fetch embeddings for the last words played by each player
            if model_name == "openai":
                embeddings_player1 = get_openai_embeddings(words_player1)
                embeddings_player2 = get_openai_embeddings(words_player2)
            else:
                embeddings_player1 = get_embeddings(words_player1, embedding_model)
                embeddings_player2 = get_embeddings(words_player2, embedding_model)

            embeddings.append({
                'gameId': row['gameId'],
                'embedding1_' + model_name: embeddings_player1,
                'embedding2_' + model_name: embeddings_player2,
            })

    embeddings_df = pd.DataFrame(embeddings)
    games_df = games_df.merge(embeddings_df, on='gameId')
    return games_df


def embedding_distance_analysis(games_df: pd.DataFrame, distance_func: callable = cosine, embedding_model: str ="openai"):
    """
    Compute and plot the cosine distances between the last words played by two players in each game.
    """
    plt.figure(figsize=(10, 5))

    # check that games_df contains the embeddings
    if 'embedding1_' + embedding_model not in games_df.columns or 'embedding2_' + embedding_model not in games_df.columns:
        raise ValueError("Embeddings not found in the DataFrame. Use the 'get_embeddings_for_table' function to fetch them.")

    # Iterate through each game
    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Analyzing Games"):
        embedding1 = eval(row['embedding1_' + embedding_model])
        embedding2 = eval(row['embedding2_' + embedding_model])

        # Ensure both players have played words
        if len(embedding1) > 0 and len(embedding2) > 0 and len(embedding1) == len(embedding2):
            distances = []
            rounds = range(len(embedding1))

            # Fetch embeddings and calculate distances for each round
            for w1, w2 in zip(embedding1, embedding2):
                distance = distance_func(w1, w2)
                distances.append(distance)

            # Plot the distances for this game
            plt.plot(rounds, distances, marker='o', linestyle='-', label=f'Game {index + 1}')

    plt.title('Cosine Distances Between Words Played by Each Player Over Rounds')
    plt.xlabel('Round Number')
    plt.ylabel('Cosine Distance')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    from scipy.spatial.distance import cosine, euclidean, cityblock, correlation
    import os

    # 1 Load the data from a database
    db_name = "merged.db"
    csv_name = "games.csv"
    # if csv_name doesn't exist we charge the db_name
    if not os.path.exists(csv_name):
        players_df, games_df = load_sql_data(db_name)
        games_df.to_csv(csv_name, index=False)

    # players_df, games_df = load_sql_data(db_name)
    games_df = pd.read_csv(csv_name)

    # 2) Get embeddings for the last words played by each player in each game
    embedding_model = "glove"
    games_df = get_embeddings_for_table(games_df, model_name=embedding_model)
    # Save the data to a csv for future use
    games_df.to_csv(csv_name, index=False)

    # Process game data to compute success rates and average rounds for winning games
    player_metrics = calculate_player_metrics(games_df)
    print("Success Rate and Average Rounds for Winning Games:")
    print(player_metrics)

    embedding_distance_analysis(games_df, distance_func=euclidean, embedding_model=embedding_model)

    # Perform embedding distance analysis and plot the results
    # embedding_distance_analysis(games_df, rounds=6)
    #
    # # Analyze strategies based on Euclidean distance calculations
    # strategy_analysis(games_df, players_df)