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

def plot_embedding_distance_during_game(games_df: pd.DataFrame, distance_func: callable = cosine, embedding_model: str ="openai"):
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

def strategy_analysis(games_df, embedding_model):
    """
    Analyze game strategy and store distance metrics as arrays
    for each game row (player perspective).

    For each row (i.e., each game a player participates in):
      - mirroring_distance: distance to the opponent's previous word each round
      - balancing_distance: distance to the player's own previous word each round
      - staying_close_distance: distance to the player's own previous word each round
      * The first element of each array is NaN (no previous word).
    """
    # Get all unique players
    players = pd.concat([games_df['player1Id'], games_df['player2Id']]).unique()

    results = []

    for player in players:
        try:
            # Select games where this player is involved
            player_games = games_df[
                (games_df['player1Id'] == player) | (games_df['player2Id'] == player)
            ].copy()

            # Assign the playerId column so each row knows which player is being analyzed
            player_games['playerId'] = player

            # Identify the embeddings for "my" words vs. "opponent" words in each game
            player_games['embedding_my'] = player_games.apply(
                lambda row: np.array(eval(row[f'embedding1_{embedding_model}']))
                if row['player1Id'] == player
                else np.array(eval(row[f'embedding2_{embedding_model}'])),
                axis=1
            )
            player_games['embedding_opponent'] = player_games.apply(
                lambda row: np.array(eval(row[f'embedding2_{embedding_model}']))
                if row['player1Id'] == player
                else np.array(eval(row[f'embedding1_{embedding_model}'])),
                axis=1
            )

            # Prepare new columns to store the arrays of strategy distances
            player_games['mirroring_distance'] = None
            player_games['balancing_distance'] = None
            player_games['staying_close_distance'] = None

            # For each game row, compute the distance arrays
            for index, game in player_games.iterrows():
                embedding_my = game['embedding_my']
                embedding_opponent = game['embedding_opponent']

                # Ensure we don't exceed the length of either embeddings array
                num_rounds = min(len(embedding_my), len(embedding_opponent))

                # Initialize empty lists to store round-by-round distances
                mirroring_list = []
                balancing_list = []
                staying_close_list = []

                for i in range(num_rounds):
                    # If i=0 => no "previous" word for either side, so store NaN
                    if i == 0:
                        mirroring_list.append(np.nan)
                        balancing_list.append(np.nan)
                        staying_close_list.append(np.nan)
                    else:
                        current_word_embed = embedding_my[i]
                        prev_opponent_word_embed = embedding_opponent[i - 1]
                        prev_my_word_embed = embedding_my[i - 1]

                        # 1) Mirroring: distance to opponent's previous word
                        mirroring_dist = cosine(current_word_embed, prev_opponent_word_embed)
                        mirroring_list.append(mirroring_dist)

                        # 2) Balancing: distance to the player's own previous word
                        balancing_dist = cosine(current_word_embed, prev_my_word_embed)
                        balancing_list.append(balancing_dist)

                        # 3) Staying close: same as balancing above?
                        staying_close_dist = cosine(current_word_embed, prev_my_word_embed)
                        staying_close_list.append(staying_close_dist)

                # Store the lists (arrays) back into the DataFrame row
                player_games.at[index, 'mirroring_distance'] = mirroring_list
                player_games.at[index, 'balancing_distance'] = balancing_list
                player_games.at[index, 'staying_close_distance'] = staying_close_list

            results.append(player_games)

        except Exception as e:
            print(f"Error processing player {player}: {e}")

    # Combine the per-player DataFrames
    return pd.concat(results, ignore_index=True)


def plot_strategy_heatmap(results_df):
    """
    Plot a heatmap showing the average distance for each strategy (mirroring,
    balancing, staying_close) per player.

    'results_df' is the DataFrame returned by 'strategy_analysis', which has:
      - playerId
      - mirroring_distance        (list of round-by-round distances)
      - balancing_distance        (list of round-by-round distances)
      - staying_close_distance    (list of round-by-round distances)
      - plus other columns like gameId, embedding_my, etc.
    """

    # The three array-type columns added in strategy_analysis
    strategy_columns = ["mirroring_distance", "balancing_distance", "staying_close_distance"]

    # Build a list of rows for a new "long" DataFrame:
    #   [playerId, strategy, distance]
    # where 'distance' is the per-game average ignoring NaNs.
    rows = []
    for idx, row in results_df.iterrows():
        player_id = row["playerId"]

        for strategy_col in strategy_columns:
            if strategy_col in row:
                round_values = row[strategy_col]

                # Ensure we have a list/array; could be None for some rows
                if isinstance(round_values, (list, np.ndarray)):
                    # Convert to float array
                    arr = np.array(round_values, dtype=float)
                    # Compute mean ignoring NaNs
                    avg_dist = np.nanmean(arr) if len(arr) > 0 else np.nan
                else:
                    # If it's not a list, just set distance to NaN
                    avg_dist = np.nan

                rows.append({
                    "playerId": player_id,
                    "strategy": strategy_col,
                    "distance": avg_dist
                })

    # Create a DataFrame of these long rows
    long_df = pd.DataFrame(rows)
    # If each (player, game) row is repeated, you can group to get
    # a single average value per (playerId, strategy) across all games.
    grouped = long_df.groupby(["playerId", "strategy"])["distance"].mean().reset_index()

    # Pivot so that each playerId is a row, each strategy is a column
    strategy_usage = grouped.pivot(index="playerId", columns="strategy", values="distance")
    strategy_usage = strategy_usage.div(strategy_usage.sum(axis=1), axis=0)

    # Ensure columns are numeric for plotting
    strategy_usage = strategy_usage.apply(pd.to_numeric, errors="coerce")

    # Plot as a heatmap using matshow
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(strategy_usage, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(cax)

    # Set tick labels
    ax.set_xticks(np.arange(len(strategy_usage.columns)))
    ax.set_yticks(np.arange(len(strategy_usage.index)))
    ax.set_xticklabels(strategy_usage.columns, rotation=90)
    ax.set_yticklabels(strategy_usage.index)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Player ID")
    ax.set_title("Average Distance per Strategy (by Player)")

    plt.show()

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

    # plot_embedding_distance_during_game(games_df, distance_func=euclidean, embedding_model=embedding_model)

    # Analyze strategies based on Euclidean distance calculations
    results_df = strategy_analysis(games_df, embedding_model)

    plot_strategy_heatmap(results_df)
