import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# External module imports
from benchmark.analysis.utils.data_loading import load_sql_data

from scipy.spatial.distance import cosine

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
        # If it's already a Python list, we can use them directly. If they're strings, we use eval:
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


def get_all_lemmas(synset):
    """
    Return a set of all lower-cased lemma names for a given synset.
    """
    return set(lemma.name().lower() for lemma in synset.lemmas())


def is_hypernym(candidate_word, original_word):
    """
    Return True if candidate_word is a hypernym of original_word.
    This function checks not only whether candidate_word's synsets
    appear in any hypernym path of original_word's synsets, but it also
    compares lemma names (to catch synonyms, adjectival forms, or morphological variants).
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()

    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    # Optionally include morphological variant(s) for candidate_word.
    candidate_morph = wn.morphy(candidate_word)
    if candidate_morph:
        candidate_forms = {candidate_word, candidate_morph.lower()}
    else:
        candidate_forms = {candidate_word}

    # Iterate over each synset of the original word.
    for orig_syn in original_synsets:
        for path in orig_syn.hypernym_paths():
            # For each synset in the hypernym path, gather all lemma names.
            path_lemmas = set()
            for syn in path:
                path_lemmas |= get_all_lemmas(syn)
            # If any candidate form is present in the hypernym path, return True.
            if candidate_forms & path_lemmas:
                return True
    return False


def is_hyponym(candidate_word, original_word):
    """
    Return True if candidate_word is a hyponym of original_word.
    This is equivalent to checking if original_word is a hypernym of candidate_word.
    """
    return is_hypernym(original_word, candidate_word)


def is_antonym(word_a, word_b):
    """
    Return True if word_b is an antonym of word_a.
    The check is case-insensitive and uses WordNet's antonym relationships.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    syns_a = wn.synsets(word_a)
    for syn in syns_a:
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                if ant.name().lower() == word_b:
                    return True
    return False


def is_synonym(word_a, word_b):
    """
    Return True if word_b is a synonym of word_a.
    This function checks if word_b appears in any of the lemma names for word_a's synsets,
    and also includes a morphological variant for word_a if available.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    syns_a = wn.synsets(word_a)
    synonyms_a = set()
    for syn in syns_a:
        for lemma in syn.lemmas():
            synonyms_a.add(lemma.name().lower())
    word_a_morph = wn.morphy(word_a)
    if word_a_morph:
        synonyms_a.add(word_a_morph.lower())
    return word_b in synonyms_a


def qualitative_analysis(player_games):
    """
    For each game (row), compute qualitative measures:
      - abstraction_measure: whether the current word is a hypernym (or its variant)
        of the opponent's previous word.
      - contrast_measure: whether the current word is an antonym of the opponent's previous word.
      - synonym_measure: whether the current word is a synonym of the player's previous word.
    The function converts boolean values to integers (1 for True, 0 for False) and stores
    a list per game.
    """
    # Prepare new columns to store the arrays of strategy measures.
    player_games['abstraction_measure'] = None
    player_games['contrast_measure'] = None
    player_games['synonym_measure'] = None

    # Iterate over each game row.
    for index, game in player_games.iterrows():
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])

        # Ensure we only iterate up to the length of the shortest word list.
        num_rounds = min(len(word_my), len(word_opponent))

        # Initialize lists to store round-by-round measures.
        abstraction_list = []
        contrast_list = []
        synonym_list = []

        for i in range(num_rounds):
            # For round 0, there is no previous word, so record NaN.
            if i == 0:
                abstraction_list.append(np.nan)
                contrast_list.append(np.nan)
                synonym_list.append(np.nan)
            else:
                current_word = word_my[i]
                prev_opponent_word = word_opponent[i - 1]
                prev_my_word = word_my[i - 1]

                # 1) Abstraction: check if current_word is a hypernym (or related variant)
                #    of the opponent's previous word.
                abstraction_score = int(is_hypernym(current_word, prev_opponent_word))
                abstraction_list.append(abstraction_score)

                # 2) Contrast: check if current_word is an antonym of the opponent's previous word.
                contrast_score = int(is_antonym(current_word, prev_opponent_word))
                contrast_list.append(contrast_score)

                # 3) Synonym: check if current_word is a synonym of the player's previous word.
                synonym_score = int(is_synonym(current_word, prev_my_word))
                synonym_list.append(synonym_score)

        # Store the lists back into the DataFrame row.
        player_games.at[index, 'abstraction_measure'] = abstraction_list
        player_games.at[index, 'contrast_measure'] = contrast_list
        player_games.at[index, 'synonym_measure'] = synonym_list

    return player_games

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



def decide_winning_strategy(row):
    """
    For each round in the game row, decide which strategy 'wins' (i.e., is marked as 1)
    and generate both:
      1. A list of winning strategy names (one per round)
      2. A list of binary encodings (one per round) where the winning strategy position is 1 and the rest are 0.

    Decision rules:
      - For round 0, no decision is possible (output NaN).
      - If any of the qualitative measures (abstraction, contrast, synonym) is 1,
        choose the winning strategy among these according to the priority:
          synonym > abstraction > contrast.
        In that case, all distance-based strategies are forced to 0.
      - Otherwise, select the distance-based strategy (mirroring, balancing, staying_close)
        with the smallest distance.

    Returns:
      A tuple (winning_strategy_names, winning_strategy_encodings), where each is a list
      of length equal to the number of rounds.
    """
    # Retrieve per-round lists for each measure.
    mirroring = row['mirroring_distance']
    balancing = row['balancing_distance']
    staying_close = row['staying_close_distance']
    abstraction = row['abstraction_measure']
    contrast = row['contrast_measure']
    synonym = row['synonym_measure']

    num_rounds = len(mirroring)  # assuming all lists have the same length

    # The fixed ordering of strategies for encoding.
    strategy_order = [
        "mirroring_distance",
        "balancing_distance",
        "staying_close_distance",
        "abstraction_measure",
        "contrast_measure",
        "synonym_measure"
    ]

    winning_strategy_names = []  # List of winning strategy names per round.
    winning_strategy_encodings = []  # List of binary arrays per round.

    for i in range(num_rounds):
        if i == 0:
            # No decision possible in round 0.
            winning_strategy_names.append(None)
            winning_strategy_encodings.append([np.nan] * len(strategy_order))
        else:
            # Get the qualitative (boolean) measures for this round.
            bool_syn = synonym[i]
            bool_abs = abstraction[i]
            bool_con = contrast[i]

            # If any qualitative measure is triggered, override the distance-based ones.
            if bool_syn == 1 or bool_abs == 1 or bool_con == 1:
                if bool_syn == 1:
                    winning_strategy = "synonym_measure"
                elif bool_abs == 1:
                    winning_strategy = "abstraction_measure"
                else:
                    winning_strategy = "contrast_measure"
            else:
                # Otherwise, choose the best among the distance-based strategies.
                distances = {
                    "mirroring_distance": mirroring[i],
                    "balancing_distance": balancing[i],
                    "staying_close_distance": staying_close[i]
                }
                winning_strategy = min(distances, key=distances.get)

            # Record the winning strategy name.
            winning_strategy_names.append(winning_strategy)

            # Build the binary encoding: for each strategy in the fixed order, mark 1 if it matches.
            encoding = [1 if strat == winning_strategy else 0 for strat in strategy_order]
            winning_strategy_encodings.append(encoding)

    return (winning_strategy_names, winning_strategy_encodings)


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
            # Compute distance-based metrics.
            player_games = quantitative_analysis(player_games)
            # Compute qualitative measures (converted to 0/1).
            player_games = qualitative_analysis(player_games)

            # For each game row, decide the winning strategy per round.
            # This returns a tuple: (winning_strategy_names, winning_strategy_encodings)
            win_strat = player_games.apply(decide_winning_strategy, axis=1)
            player_games['winning_strategy_name'] = win_strat.apply(lambda x: x[0])
            player_games['winning_strategy_encoding'] = win_strat.apply(lambda x: x[1])

            results.append(player_games)

        except Exception as e:
            print(f"Error processing player {player}: {e}")

    # Combine the per-player DataFrames
    return pd.concat(results, ignore_index=True)




def plot_strategy_heatmap(results_df):
    """
    Plot a heatmap showing the average winning strategy for each player,
    using the winning_strategy_encoding column.

    Each entry in the 'winning_strategy_encoding' column is a list (over rounds)
    of binary vectors (length=6) following the fixed order:
       [mirroring_distance, balancing_distance, staying_close_distance,
        abstraction_measure, contrast_measure, synonym_measure]

    The heatmap shows, for each player, the fraction of rounds in which each strategy was the winner.
    Each cell is annotated with the percentage value, and the colorbar is formatted in percentages.
    """
    # Define the fixed ordering of strategies.
    strategy_order = [
        "mirroring_distance",
        "balancing_distance",
        "staying_close_distance",
        "abstraction_measure",
        "contrast_measure",
        "synonym_measure"
    ]

    # Build a long-form DataFrame: each row = (playerId, strategy, average_value)
    rows = []
    for idx, row in results_df.iterrows():
        player_id = row["playerId"]
        encoding_list = row.get("winning_strategy_encoding", None)

        if encoding_list is not None and isinstance(encoding_list, (list, np.ndarray)):
            # Convert the list of binary vectors to a NumPy array.
            # Each inner vector should be of length 6.
            arr = np.array(encoding_list, dtype=float)  # shape: (num_rounds, 6)
            # Compute the average over rounds (ignoring NaNs)
            avg_vals = np.nanmean(arr, axis=0)  # shape: (6,)
        else:
            avg_vals = np.full(len(strategy_order), np.nan)

        # For each strategy (by fixed order), record its average value.
        for i, strategy in enumerate(strategy_order):
            rows.append({
                "playerId": player_id,
                "strategy": strategy,
                "value": avg_vals[i]
            })

    long_df = pd.DataFrame(rows)
    # If a player played multiple games, average across games.
    grouped = long_df.groupby(["playerId", "strategy"])["value"].mean().reset_index()
    # Pivot: rows = playerId, columns = strategies.
    strategy_usage = grouped.pivot(index="playerId", columns="strategy", values="value")

    # Create the heatmap.
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(strategy_usage.values,
                     cmap="coolwarm",
                     aspect="auto",
                     interpolation="nearest")

    # Add a colorbar with percentage formatting.
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Set tick labels: strategies on x-axis, player IDs on y-axis.
    ax.set_xticks(np.arange(len(strategy_usage.columns)))
    ax.set_yticks(np.arange(len(strategy_usage.index)))
    ax.set_xticklabels(strategy_usage.columns, rotation=90)
    ax.set_yticklabels(strategy_usage.index)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Player ID")
    ax.set_title("Average Winning Strategy Measures by Player (Percentage)")

    # Annotate each cell with percentage values.
    for i in range(strategy_usage.shape[0]):
        for j in range(strategy_usage.shape[1]):
            val = strategy_usage.values[i, j]
            if not np.isnan(val):
                # Multiply by 100 to show a percentage.
                ax.text(j, i, f"{val * 100:.0f}%", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


def plot_strategy_heatmap_human_vs_bot(results_df):
    """
    Plot a heatmap showing the average winning strategy for each game configuration,
    using the winning_strategy_encoding column.

    Each entry in the 'winning_strategy_encoding' column is a list (over rounds)
    of binary vectors (length=6) following the fixed order:
       [mirroring_distance, balancing_distance, staying_close_distance,
        abstraction_measure, contrast_measure, synonym_measure]

    The heatmap shows, for each game configuration (Human vs Bot or Human vs Human),
    the fraction of rounds in which each strategy was the winner.
    Each cell is annotated with the percentage value, and the colorbar is formatted in percentages.
    """
    # Define the fixed ordering of strategies.
    strategy_order = [
        "mirroring_distance",
        "balancing_distance",
        "staying_close_distance",
        "abstraction_measure",
        "contrast_measure",
        "synonym_measure"
    ]

    # Build a long-form DataFrame: each row = (game_type, strategy, average_value)
    rows = []
    for idx, row in results_df.iterrows():
        # Determine game configuration using the 'BotId' column.
        if "botId" in row:
            if pd.isna(row["botId"]) or row["botId"] == "":
                game_type = "Human vs Human"
            else:
                game_type = "Human vs Bot"
        else:
            game_type = "Unknown"

        encoding_list = row.get("winning_strategy_encoding", None)

        if encoding_list is not None and isinstance(encoding_list, (list, np.ndarray)):
            # Convert the list of binary vectors to a NumPy array.
            # Each inner vector should be of length 6.
            arr = np.array(encoding_list, dtype=float)  # shape: (num_rounds, 6)
            # Compute the average over rounds (ignoring NaNs)
            avg_vals = np.nanmean(arr, axis=0)  # shape: (6,)
        else:
            avg_vals = np.full(len(strategy_order), np.nan)

        # For each strategy (by fixed order), record its average value.
        for i, strategy in enumerate(strategy_order):
            rows.append({
                "game_type": game_type,
                "strategy": strategy,
                "value": avg_vals[i]
            })

    long_df = pd.DataFrame(rows)

    # Group by game configuration and strategy, averaging over games.
    grouped = long_df.groupby(["game_type", "strategy"])["value"].mean().reset_index()

    # Pivot so that rows = game configuration, columns = strategies.
    strategy_usage = grouped.pivot(index="game_type", columns="strategy", values="value")

    # Create the heatmap.
    fig, ax = plt.subplots(figsize=(8, 4))
    cax = ax.matshow(strategy_usage.values,
                     cmap="coolwarm",
                     aspect="auto",
                     interpolation="nearest")

    # Add a colorbar with percentage formatting.
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Set tick labels: strategies on x-axis, game configuration on y-axis.
    ax.set_xticks(np.arange(len(strategy_usage.columns)))
    ax.set_yticks(np.arange(len(strategy_usage.index)))
    ax.set_xticklabels(strategy_usage.columns, rotation=90)
    ax.set_yticklabels(strategy_usage.index)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Game Configuration")
    ax.set_title("Average Winning Strategy Measures by Game Configuration (Percentage)")

    # Annotate each cell with percentage values.
    for i in range(strategy_usage.shape[0]):
        for j in range(strategy_usage.shape[1]):
            val = strategy_usage.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val * 100:.0f}%", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

def print_game_turns(results_df, n=20):
    """
    For the first n games in results_df, print for each turn (except turn 0):
      - The words from the previous turn (player's and opponent's)
      - An arrow ("->")
      - The player's current turn word
      - The winning strategy name for that turn

    Assumes that each row in results_df has the following columns:
      - 'word_my': list of words played by the player (in order)
      - 'word_opponent': list of words played by the opponent (in order)
      - 'winning_strategy_name': list of winning strategy names per round (with round 0 as None)
    """
    # Select only the first n games
    for game_idx, row in results_df.head(n).iterrows():
        print(f"Game {game_idx}:")
        word_my = eval(row['word_my'])
        word_opponent = eval(row['word_opponent'])
        winning_strats = row['winning_strategy_name']

        # Determine the number of rounds for this game (based on the minimum length of the lists)
        num_rounds = min(len(word_my), len(word_opponent), len(winning_strats))
        if num_rounds < 2:
            print("  Not enough rounds to display details.")
            continue

        # For each turn except the first one, print the desired info.
        for i in range(1, num_rounds):
            prev_player_word = word_my[i - 1]
            prev_opponent_word = word_opponent[i - 1]
            current_word = word_my[i]
            winning_strategy = winning_strats[i]

            print(
                f"  Turn {i}: [{prev_player_word} / {prev_opponent_word}] -> {current_word}  (winning strategy: {winning_strategy})")
        print()  # Blank line between games



if __name__ == "__main__":
    import os
    from scipy.spatial.distance import cosine
    from benchmark.analysis.utils.embeding_utils import get_embeddings_for_table, calculate_pca_for_embeddings
    from game_statistics import calculate_game_metrics_per_player

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
    plot_strategy_heatmap_human_vs_bot(results_df)
    # plot_strategy_heatmap(results_df)

    # print_game_turns(results_df, n=20)
