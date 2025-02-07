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

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def is_hypernym(candidate_word, original_word):
    """
    Return True if candidate_word is a hypernym of original_word.
    That is, if candidate_word appears in any hypernym path of original_word.
    """
    candidate_synsets = wn.synsets(candidate_word.lower())
    original_synsets = wn.synsets(original_word.lower())
    for orig_syn in original_synsets:
        for path in orig_syn.hypernym_paths():
            if any(c_syn in path for c_syn in candidate_synsets):
                return True
    return False



def is_hyponym(candidate_word, original_word):
    """
    Return True if `candidate_word` is a hyponym of `original_word`.
    That is effectively the same as asking if `original_word` is a hypernym of `candidate_word`.
    """
    return is_hypernym(original_word, candidate_word)


def is_antonym(word_a, word_b):
    syns_a = wn.synsets(word_a)
    for syn in syns_a:
        for lemma in syn.lemmas():
            if lemma.antonyms():
                # collect lemma.antonyms()
                if word_b in [ant.name() for ant in lemma.antonyms()]:
                    return True
    return False

def is_synonym(word_a, word_b):
    syns_a = wn.synsets(word_a)
    synonyms_a = set()
    for syn in syns_a:
        for lemma in syn.lemmas():
            synonyms_a.add(lemma.name())
    return word_b in synonyms_a

def quantitative_analysis(player_games):
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

                # 2) Balancing: distance to the average of the two previous words
                balancing_dist = cosine(current_word_embed, (prev_my_word_embed + prev_opponent_word_embed) / 2)
                balancing_list.append(balancing_dist)

                # 3) Staying close: distance to the player's own previous word
                staying_close_dist = cosine(current_word_embed, prev_my_word_embed)
                staying_close_list.append(staying_close_dist)

        # Store the lists (arrays) back into the DataFrame row
        player_games.at[index, 'mirroring_distance'] = mirroring_list
        player_games.at[index, 'balancing_distance'] = balancing_list
        player_games.at[index, 'staying_close_distance'] = staying_close_list

    return player_games

def qualitative_analysis(player_games):
    # Prepare new columns to store the arrays of strategy distances
    player_games['abstraction_measure'] = None
    player_games['contrast_measure'] = None
    player_games['synonym_measure'] = None

    # For each game row, compute measure
    for index, game in player_games.iterrows():
        word_my = game['word_my']
        word_opponent = game['word_opponent']

        # Ensure we don't exceed the length of either embeddings array
        num_rounds = min(len(word_my), len(word_opponent))

        # Initialize empty lists to store round-by-round distances
        abstraction_list = []
        contrast_list = []
        synonym_list = []

        for i in range(num_rounds):
            # If i=0 => no "previous" word for either side, so store NaN
            if i == 0:
                abstraction_list.append(np.nan)
                contrast_list.append(np.nan)
                synonym_list.append(np.nan)
            else:
                current_word = word_my[i]
                prev_opponent_word = word_opponent[i - 1]
                prev_my_word = word_my[i - 1]

                # 1) Abstraction: check if the current word is a hypernym of the opponent's previous word
                abstraction_score = int(is_hypernym(current_word, prev_opponent_word))
                abstraction_list.append(abstraction_score)

                # 2) Contrast: check if the current word is an antonym of the opponent's previous word
                contrast_score = int(is_antonym(current_word, prev_opponent_word))
                contrast_list.append(contrast_score)

                # 3) Synonym: check if the current word is a synonym of the player's previous word
                synonym_score = int(is_synonym(current_word, prev_my_word))
                synonym_list.append(synonym_score)

        # Store the lists (arrays) back into the DataFrame row
        player_games.at[index, 'abstraction_measure'] = abstraction_list
        player_games.at[index, 'contrast_measure'] = contrast_list
        player_games.at[index, 'synonym_measure'] = synonym_list

    return player_games


def decide_winning_strategy(row):
    """
    For each round, decide which strategy 'wins' (assign 1) and mark all others as 0.

    Decision rule:
      - If any of the qualitative measures (abstraction, contrast, synonym) is 1,
        then choose one of them based on the following priority:
            Priority: synonym > abstraction > contrast.
        In that case, all distance-based strategies (mirroring, balancing, staying_close)
        are forced to 0.
      - Otherwise, choose the distance-based strategy with the minimal distance
        (i.e. the smallest value among mirroring, balancing, and staying_close).

    For round 0, no decision is made (values remain NaN).
    """
    # Retrieve the per-round lists for each strategy measure.
    mirroring = row['mirroring_distance']
    balancing = row['balancing_distance']
    staying_close = row['staying_close_distance']
    abstraction = row['abstraction_measure']
    contrast = row['contrast_measure']
    synonym = row['synonym_measure']

    num_rounds = len(mirroring)  # Assume all lists have equal length.

    # Initialize new binary lists for each strategy.
    mirroring_win = []
    balancing_win = []
    staying_close_win = []
    abstraction_win = []
    contrast_win = []
    synonym_win = []

    for i in range(num_rounds):
        if i == 0:
            # For round 0, there's no previous round to compare with.
            mirroring_win.append(np.nan)
            balancing_win.append(np.nan)
            staying_close_win.append(np.nan)
            abstraction_win.append(np.nan)
            contrast_win.append(np.nan)
            synonym_win.append(np.nan)
        else:
            # Get the qualitative (boolean) measures for this round.
            bool_syn = synonym[i]
            bool_abs = abstraction[i]
            bool_con = contrast[i]

            # If any qualitative measure is triggered, choose from them.
            if bool_syn == 1 or bool_abs == 1 or bool_con == 1:
                # Qualitative measures override distance-based ones.
                # Choose based on a fixed priority: synonym > abstraction > contrast.
                if bool_syn == 1:
                    winning_strategy = 'synonym'
                elif bool_abs == 1:
                    winning_strategy = 'abstraction'
                else:
                    winning_strategy = 'contrast'
            else:
                # Otherwise, choose the distance-based strategy with the minimal distance.
                distances = {
                    'mirroring': mirroring[i],
                    'balancing': balancing[i],
                    'staying_close': staying_close[i]
                }
                winning_strategy = min(distances, key=distances.get)

            # Set the winning strategy to 1 and all others to 0.
            mirroring_win.append(1 if winning_strategy == 'mirroring' else 0)
            balancing_win.append(1 if winning_strategy == 'balancing' else 0)
            staying_close_win.append(1 if winning_strategy == 'staying_close' else 0)
            abstraction_win.append(1 if winning_strategy == 'abstraction' else 0)
            contrast_win.append(1 if winning_strategy == 'contrast' else 0)
            synonym_win.append(1 if winning_strategy == 'synonym' else 0)

    # Return a dictionary mapping each strategy column to its new binary list.
    return {
        'mirroring_distance': mirroring_win,
        'balancing_distance': balancing_win,
        'staying_close_distance': staying_close_win,
        'abstraction_measure': abstraction_win,
        'contrast_measure': contrast_win,
        'synonym_measure': synonym_win
    }

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
            player_games['word_my'] = player_games.apply(
                lambda row: row['wordsPlayed1'] if row['player1Id'] == player else row['wordsPlayed2'],
                axis=1
            )
            player_games['word_opponent'] = player_games.apply(
                lambda row: row['wordsPlayed2'] if row['player1Id'] == player else row['wordsPlayed1'],
                axis=1
            )
            player_games = quantitative_analysis(player_games)
            player_games = qualitative_analysis(player_games)
            # For each game row, decide the winning strategy per round.
            win_strat = player_games.apply(decide_winning_strategy, axis=1)
            # Update the relevant columns with the new binary lists.
            for col in ['mirroring_distance', 'balancing_distance', 'staying_close_distance',
                        'abstraction_measure', 'contrast_measure', 'synonym_measure']:
                player_games[col] = win_strat.apply(lambda x: x[col])


            results.append(player_games)

        except Exception as e:
            print(f"Error processing player {player}: {e}")

    # Combine the per-player DataFrames
    return pd.concat(results, ignore_index=True)


def plot_strategy_heatmap(results_df):
    """
    Plot a heatmap showing the average value for each strategy-related column
    (both distance-based and boolean-based) per player.

    The DataFrame 'results_df' is expected to contain columns:
      - playerId
      - mirroring_distance, balancing_distance, staying_close_distance  (lists of floats)
      - abstraction_measure, contrast_measure, synonym_measure          (lists of booleans)
      - other columns like gameId, embedding_my, word_my, etc.
    """

    # 1) List **all** columns you want to plot:
    #    - distance-based:   mirroring_distance, balancing_distance, staying_close_distance
    #    - boolean-based:    abstraction_measure, contrast_measure, synonym_measure
    strategy_columns = [
        "mirroring_distance",
        "balancing_distance",
        "staying_close_distance",
        "abstraction_measure",
        "contrast_measure",
        "synonym_measure"
    ]

    # 2) Build a long-form DataFrame: each row = (playerId, strategy, average_value)
    rows = []
    for idx, row in results_df.iterrows():
        player_id = row["playerId"]

        for strategy_col in strategy_columns:
            if strategy_col in row:
                round_values = row[strategy_col]  # This should be a list of values

                if isinstance(round_values, (list, np.ndarray)):
                    # Convert booleans or whatever type to floats
                    arr = np.array(round_values, dtype=float)
                    # Compute the average ignoring NaNs
                    #  - For distances, mean distance
                    #  - For boolean columns, mean = fraction of rounds that are True
                    avg_val = np.nanmean(arr) if len(arr) > 0 else np.nan
                else:
                    avg_val = np.nan

                rows.append({
                    "playerId": player_id,
                    "strategy": strategy_col,
                    "value": avg_val  # rename to 'value' for clarity
                })

    long_df = pd.DataFrame(rows)

    # 3) Aggregate if multiple games exist per player
    grouped = long_df.groupby(["playerId", "strategy"])["value"].mean().reset_index()

    # 4) Pivot: rows = playerId, columns = strategies
    strategy_usage = grouped.pivot(index="playerId", columns="strategy", values="value")

    # If you want raw average values (distance or fraction) in the heatmap, DO NOT normalize by row sum.
    # If you do want to see each player's distribution of strategies, you could do:
    # strategy_usage = strategy_usage.div(strategy_usage.sum(axis=1), axis=0)
    # But that mixes distances and boolean fractions in an odd way.

    # 5) Plot as a heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Matshow expects a 2D numpy array
    # We can supply .values but be mindful of NaNs
    cax = ax.matshow(strategy_usage.values,
                     cmap="coolwarm",
                     aspect="auto",
                     interpolation="nearest")

    # Add colorbar
    fig.colorbar(cax)

    # Set tick labels (strategies on x-axis, player IDs on y-axis)
    ax.set_xticks(range(len(strategy_usage.columns)))
    ax.set_yticks(range(len(strategy_usage.index)))
    ax.set_xticklabels(strategy_usage.columns, rotation=90)
    ax.set_yticklabels(strategy_usage.index)

    ax.set_xlabel("Strategy / Measure")
    ax.set_ylabel("Player ID")
    ax.set_title("Average Strategy Measures by Player")

    plt.tight_layout()
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
    # plot_embedding_distance_during_game(
    #     games_df,
    #     distance_func=cosine,
    #     embedding_model="glove",
    #     use_pca=True
    # )
    # plot_embedding_distance_during_game(
    #     games_df,
    #     distance_func=cosine,
    #     embedding_model="glove",
    #     use_pca=False
    # )
    # 5) Strategy analysis (using the PCA columns):
    results_df = strategy_analysis(games_df, embedding_model, use_pca=True)
    plot_strategy_heatmap(results_df)