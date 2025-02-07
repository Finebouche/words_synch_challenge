import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

# External module imports
from data_loading import load_sql_data

from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA


def calculate_game_metrics_per_player(games_df):
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


def calculate_pca_for_embeddings(games_df: pd.DataFrame, model_name="openai", num_pca_components=None):

    embed_col1 = f"embedding1_{model_name}"
    embed_col2 = f"embedding2_{model_name}"

    # Check that the embeddings columns exist
    if embed_col1 not in games_df.columns or embed_col2 not in games_df.columns:
        raise ValueError(
            f"Embeddings not found in the DataFrame. Columns '{embed_col1}' or '{embed_col2}' missing. "
            f"Make sure you ran 'get_embeddings_for_table' first."
        )
    #    We create new columns: e.g. 'embedding1_glove_pca'
    new_col1 = f"{embed_col1}_pca"
    new_col2 = f"{embed_col2}_pca"

    # If user specifies a number of PCA components, reduce the dimension.
    if num_pca_components is not None:
        print(f"Performing PCA to reduce embeddings to {num_pca_components} dimensions.")

        # 1) Collect *all* round-level embeddings across all games for both players
        #    in a big list. We'll store them separately as:
        all_vectors = []  # shape: (N * R, D)  where R is #rounds in a game, D is original dimension

        # We'll also need to keep an index (game, 'player1'/'player2', round_id)
        # so we can reconstruct the data after PCA transform
        index_info = []

        for idx, row in games_df.iterrows():
            emb1 = eval(row.get(embed_col1, []))
            emb2 = eval(row.get(embed_col2, []))

            # Convert to np.array if not empty
            emb1_arr = np.array(emb1, dtype=float) if len(emb1) > 0 else np.empty((0, 0))
            emb2_arr = np.array(emb2, dtype=float) if len(emb2) > 0 else np.empty((0, 0))

            # Player1
            for r_i in range(emb1_arr.shape[0]):
                all_vectors.append(emb1_arr[r_i])
                index_info.append((idx, 'player1', r_i))
            # Player2
            for r_i in range(emb2_arr.shape[0]):
                all_vectors.append(emb2_arr[r_i])
                index_info.append((idx, 'player2', r_i))

        # Convert all_vectors to numpy array
        if len(all_vectors) > 0:
            all_vectors_np = np.array(all_vectors, dtype=float)
        else:
            all_vectors_np = np.empty((0, 0))

        # 2) Fit PCA
        if all_vectors_np.shape[0] > 0:
            pca = PCA(n_components=num_pca_components)
            pca.fit(all_vectors_np)

            # 3) Transform all vectors
            transformed_embeddings = pca.transform(all_vectors_np)

            # 4) Rebuild them into lists-of-rounds for each row/player
            # We'll store the new columns in memory and then assign to DataFrame at the end
            # to avoid repeated rewriting of rows.
            new_emb1_series = [[] for _ in range(len(games_df))]
            new_emb2_series = [[] for _ in range(len(games_df))]

            for (df_idx, player, round_idx), pca_vec in zip(index_info, transformed_embeddings):
                if player == 'player1':
                    new_emb1_series[df_idx].append(pca_vec.tolist())
                else:
                    new_emb2_series[df_idx].append(pca_vec.tolist())

            # Now save these lists into games_df
            games_df[new_col1] = new_emb1_series
            games_df[new_col2] = new_emb2_series

        else:
            # If no data, just create empty columns
            new_col1 = f"{embed_col1}_pca"
            new_col2 = f"{embed_col2}_pca"
            games_df[new_col1] = [[] for _ in range(len(games_df))]
            games_df[new_col2] = [[] for _ in range(len(games_df))]

    return games_df


def plot_embedding_distance_during_game(games_df: pd.DataFrame,
                                        distance_func: callable = cosine,
                                        embedding_model: str = "openai",
                                        use_pca: bool = False):
    """
    Compute and plot the distance (by default, cosine) between the last words
    played by two players in each game (round by round).

    :param games_df: The dataframe containing game data, including embeddings.
    :param distance_func: The distance function to use (e.g., cosine, euclidean).
    :param embedding_model: The base name of the embedding columns (e.g. 'openai', 'glove').
    :param use_pca: If True, use the PCA-reduced embeddings (i.e., '..._pca' columns).
    """

    # Decide which columns to use
    col1 = f"embedding1_{embedding_model}"
    col2 = f"embedding2_{embedding_model}"
    if use_pca:
        col1 += "_pca"
        col2 += "_pca"

    # Check if the columns exist
    if col1 not in games_df.columns or col2 not in games_df.columns:
        raise ValueError(
            f"Embeddings not found in the DataFrame. Columns '{col1}' or '{col2}' missing. "
            f"Make sure you ran 'get_embeddings_for_table' with PCA if use_pca=True."
        )

    plt.figure(figsize=(10, 5))

    # Iterate through each game
    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Analyzing Games"):
        # Depending on how data is stored, we might need to parse strings to lists.
        # If it's already a Python list, you can use them directly. If they're strings, we use eval:
        if isinstance(row[col1], list):
            embedding1 = row[col1]
            embedding2 = row[col2]
        else:
            embedding1 = eval(row[col1])
            embedding2 = eval(row[col2])

        # Ensure both players have embedding lists and they're the same length
        if (len(embedding1) > 0 and len(embedding2) > 0 and len(embedding1) == len(embedding2)):

            distances = []
            rounds = range(len(embedding1))

            for w1, w2 in zip(embedding1, embedding2):
                # Convert to numpy just in case
                w1_arr = np.array(w1, dtype=float)
                w2_arr = np.array(w2, dtype=float)
                distances.append(distance_func(w1_arr, w2_arr))

            # Plot the distances for this game
            plt.plot(rounds, distances, marker='o', linestyle='-', label=f'Game {row["gameId"]}')

    plt.title(f'{distance_func.__name__.capitalize()} Distance Over Rounds\n'
              f'({"PCA" if use_pca else "Original"}) - Embeddings: {embedding_model}')
    plt.xlabel('Round Number')
    plt.ylabel(f'{distance_func.__name__.capitalize()} Distance')
    plt.legend()
    plt.grid(True)
    plt.show()


def strategy_analysis(games_df, embedding_model, use_pca=False):
    """
    Analyze game strategy and store distance metrics as arrays
    for each game row (player perspective).
    """
    # Decide which embedding columns to use
    emb_col1 = f"embedding1_{embedding_model}"
    emb_col2 = f"embedding2_{embedding_model}"
    if use_pca:
        emb_col1 += "_pca"
        emb_col2 += "_pca"

    # Get all unique players
    players = pd.concat([games_df['player1Id'], games_df['player2Id']]).unique()

    results = []

    for player in players:
        try:
            # Select games where this player is involved
            player_games = games_df[
                (games_df['player1Id'] == player) | (games_df['player2Id'] == player)
                ].copy()

            player_games['playerId'] = player

            player_games['embedding_my'] = player_games.apply(
                lambda row: np.array(row[emb_col1], dtype=float)
                if row['player1Id'] == player
                else np.array(row[emb_col2], dtype=float),
                axis=1
            )
            player_games['embedding_opponent'] = player_games.apply(
                lambda row: np.array(row[emb_col2], dtype=float)
                if row['player1Id'] == player
                else np.array(row[emb_col1], dtype=float),
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
    import os
    from scipy.spatial.distance import cosine, euclidean, cityblock, correlation
    from embeding_utils import get_embeddings_for_table

    db_name = "merged.db"
    csv_name = "games.csv"

    # 1) Load the data
    if not os.path.exists(csv_name):
        players_df, games_df = load_sql_data(db_name)
        games_df.to_csv(csv_name, index=False)
    else:
        games_df = pd.read_csv(csv_name)

    # 2) Get embeddings (and do PCA with e.g. 50 components)
    embedding_model = "glove"
    games_df = get_embeddings_for_table( games_df, model_name=embedding_model,)

    game_df = calculate_pca_for_embeddings(
        games_df,
        model_name=embedding_model,
        num_pca_components=15,
    )

    # Save to CSV for future use
    games_df.to_csv(csv_name, index=False)

    # 3) Calculate player metrics
    player_metrics = calculate_game_metrics_per_player(games_df)
    print("Success Rate and Average Rounds for Winning Games:")
    print(player_metrics)

    # 4) Plot distances with the original or PCA embeddings
    plot_embedding_distance_during_game(
        games_df,
        distance_func=cosine,
        embedding_model="glove",
        use_pca=True
    )
    plot_embedding_distance_during_game(
        games_df,
        distance_func=cosine,
        embedding_model="glove",
        use_pca=False
    )
    # 5) Strategy analysis (using the PCA columns):
    results_df = strategy_analysis(games_df, embedding_model, use_pca=True)
    plot_strategy_heatmap(results_df)