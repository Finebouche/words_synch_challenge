import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# External module imports
from benchmark.analysis.utils.data_loading import load_sql_data


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


#########################
# WORDNET RELATIONSHIP  #
#########################

def is_hypernym(candidate_word, original_word):
    """
    Return True if any synset of `candidate_word` appears in
    the hypernym paths of any synset of `original_word`.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    for o_syn in original_synsets:
        for path in o_syn.hypernym_paths():
            if any(c_syn in path for c_syn in candidate_synsets):
                return True
    return False

def is_hyponym(candidate_word, original_word):
    """
    Return True if candidate_word is a hyponym of original_word.
    i.e., original_word is a hypernym of candidate_word.
    """
    return is_hypernym(original_word, candidate_word)

def is_antonym(word_a, word_b):
    """
    Return True if word_b is an antonym of word_a.
    Uses WordNet antonym relationships.
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

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger_eng')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """
    Convert POS tag from the Penn Treebank or similar to a
    simplified WordNet tag: (n)oun, (v)erb, (a)djective, (r)adverb.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def is_morphological_variation(word_a, word_b):
    # Step 1: POS-tag the words
    tokens = [word_a, word_b]
    tagged = nltk.pos_tag(tokens)  # e.g. [('stronger', 'JJR'), ('strong', 'JJ')]

    w_a, pos_a = tagged[0]
    w_b, pos_b = tagged[1]

    # Step 2: Convert the Treebank tag to a WordNet tag
    wn_pos_a = get_wordnet_pos(pos_a) or wordnet.NOUN  # fallback = NOUN
    wn_pos_b = get_wordnet_pos(pos_b) or wordnet.NOUN

    # Step 3: Lemmatize with the correct POS
    lem_a = lemmatizer.lemmatize(w_a.lower(), wn_pos_a)
    lem_b = lemmatizer.lemmatize(w_b.lower(), wn_pos_b)

    # Step 4: Check if they share the same lemma, but differ as strings
    return (lem_a == lem_b)

def is_synonym(word_a, word_b):
    """
    Return True if word_a and word_b share at least one synset in WordNet.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    synsets_a = set(wn.synsets(word_a))
    synsets_b = set(wn.synsets(word_b))
    # If there's any overlap, it indicates a shared sense (synonym in at least one sense).
    return len(synsets_a.intersection(synsets_b)) > 0

###########################
# THEMATIC ALIGNMENT TEST #
###########################

def share_broad_category(word_a, word_b, depth_threshold=2):
    """
    Returns True if `word_a` and `word_b` share a common hypernym
    within `depth_threshold` levels up the hypernym tree.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    syns_a = wn.synsets(word_a)
    syns_b = wn.synsets(word_b)

    for syn_a in syns_a:
        paths_a = syn_a.hypernym_paths()
        up_a = set()
        for path in paths_a:
            for i in range(len(path)-1, max(-1, len(path)-1 - depth_threshold), -1):
                up_a.add(path[i])

        for syn_b in syns_b:
            paths_b = syn_b.hypernym_paths()
            up_b = set()
            for path in paths_b:
                for i in range(len(path)-1, max(-1, len(path)-1 - depth_threshold), -1):
                    up_b.add(path[i])

            if up_a.intersection(up_b):
                return True
    return False

def is_thematic_alignment(word_a, word_b):
    return share_broad_category(word_a, word_b, depth_threshold=2)

##############################
#  QUALITATIVE (BOOLEAN)    #
##############################

def qualitative_analysis(player_games):
    """
    Compute 3 boolean-based measures per round:
      - abstraction_measure: is current_word a hypernym of opponent's prev word?
      - contrast_measure: is current_word an antonym of opponent's prev word?
      - synonym_measure: is current_word a synonym of player's own prev word?

    Stores results in columns:
      - 'abstraction_measure'
      - 'contrast_measure'
      - 'synonym_measure'
      - 'morhological_variation_measure'
    Each is a list of length = #rounds with 0/1 or np.nan for round 0.
    """
    player_games['abstraction_measure'] = None
    player_games['contrast_measure'] = None
    player_games['synonym_measure'] = None
    player_games['morphological_variation_measure'] = None

    for index, game in player_games.iterrows():
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])
        num_rounds = min(len(word_my), len(word_opponent))

        abstraction_list = []
        contrast_list = []
        synonym_list = []
        morph_variation_list = []

        for i in range(num_rounds):
            if i == 0:
                abstraction_list.append(np.nan)
                contrast_list.append(np.nan)
                synonym_list.append(np.nan)
                morph_variation_list.append(np.nan)
            else:
                current_word = word_my[i]
                prev_opponent_word = word_opponent[i - 1]
                prev_my_word = word_my[i - 1]

                abstraction_score = int(is_hypernym(current_word, prev_opponent_word))
                contrast_score = int(is_antonym(current_word, prev_opponent_word))
                synonym_score = int(is_synonym(current_word, prev_my_word))
                morph_variation_score = int(is_morphological_variation(current_word, prev_my_word))

                abstraction_list.append(abstraction_score)
                contrast_list.append(contrast_score)
                synonym_list.append(synonym_score)
                morph_variation_list.append(morph_variation_score)

        player_games.at[index, 'abstraction_measure'] = abstraction_list
        player_games.at[index, 'contrast_measure'] = contrast_list
        player_games.at[index, 'synonym_measure'] = synonym_list
        player_games.at[index, 'morphological_variation_measure'] = morph_variation_list

    return player_games

##############################
# QUANTITATIVE (DISTANCES)  #
##############################


def expansion_score(vec_new, vec_a, vec_b):
    """
    Returns the cosine similarity between vec_new and the average of vec_a + vec_b.
    Higher => more of a 'bridge' or expansion between a & b.
    """
    from numpy.linalg import norm
    avg_vec = (vec_a + vec_b) / 2.0
    dot = np.dot(vec_new, avg_vec)
    sim = dot / (norm(vec_new)*norm(avg_vec) + 1e-9)
    return sim

def min_max_normalize(values):
    """
    Given a list of numeric values (with possible np.nan),
    perform min–max normalization so that the non-nan values fall in [0, 1].
    If all values are nan or constant, returns a list with the original values.
    """
    arr = np.array(values, dtype=float)
    # Identify valid (non-NaN) values:
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return values
    vmin = np.min(arr[mask])
    vmax = np.max(arr[mask])
    # Avoid division by zero: if constant, return zeros for non-nan entries.
    if vmax == vmin:
        return [0 if not np.isnan(v) else np.nan for v in arr]
    norm = (arr - vmin) / (vmax - vmin)
    return norm.tolist()

def quantitative_analysis(player_games):
    """
    Computes round-by-round numeric distances or similarities:
      - mirroring_distance: cosine distance to opponent's previous word's embedding
      - balancing_distance: cosine distance to average(own_prev, opp_prev)
      - staying_close_distance: cosine distance to own previous word
      - conceptual_expansion_distance: similarity (later inverted to act like a distance)
    Also computes normalized versions of these measures (per game).
    """
    player_games['mirroring_distance'] = None
    player_games['balancing_distance'] = None
    player_games['staying_close_distance'] = None
    player_games['conceptual_expansion_distance'] = None

    for index, game in player_games.iterrows():
        embedding_my = game['embedding_my']
        embedding_opponent = game['embedding_opponent']
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])

        num_rounds = min(len(embedding_my), len(embedding_opponent), len(word_my), len(word_opponent))

        mirroring_list = []
        balancing_list = []
        staying_close_list = []
        expansion_list = []

        for i in range(num_rounds):
            if i == 0:
                mirroring_list.append(np.nan)
                balancing_list.append(np.nan)
                staying_close_list.append(np.nan)
                expansion_list.append(np.nan)
            else:
                current_word_embed = embedding_my[i]
                prev_opp_embed = embedding_opponent[i - 1]
                prev_my_embed = embedding_my[i - 1]

                # Compute cosine distances (lower is "closer")
                mirroring_dist = cosine(current_word_embed, prev_opp_embed)
                balancing_dist = cosine(current_word_embed, (prev_my_embed + prev_opp_embed) / 2)
                staying_close_dist = cosine(current_word_embed, prev_my_embed)

                mirroring_list.append(mirroring_dist)
                balancing_list.append(balancing_dist)
                staying_close_list.append(staying_close_dist)

                # Conceptual expansion: computed as cosine similarity.
                sim_expansion = expansion_score(current_word_embed, prev_my_embed, prev_opp_embed)
                # Convert similarity to a distance measure:
                expansion_distance = 1.0 - sim_expansion
                expansion_list.append(expansion_distance)

        # Store raw results:
        player_games.at[index, 'mirroring_distance'] = mirroring_list
        player_games.at[index, 'balancing_distance'] = balancing_list
        player_games.at[index, 'staying_close_distance'] = staying_close_list
        player_games.at[index, 'conceptual_expansion_distance'] = expansion_list

        # Normalize each measure for this game (per row normalization across rounds)
        norm_mirroring = min_max_normalize(mirroring_list)
        norm_balancing = min_max_normalize(balancing_list)
        norm_staying_close = min_max_normalize(staying_close_list)
        norm_expansion = min_max_normalize(expansion_list)

        player_games.at[index, 'mirroring_distance'] = norm_mirroring
        player_games.at[index, 'balancing_distance'] = norm_balancing
        player_games.at[index, 'staying_close_distance'] = norm_staying_close
        player_games.at[index, 'conceptual_expansion_distance'] = norm_expansion

    return player_games

##################################
#  MAIN STRATEGY ANALYSIS LOOP   #
##################################

def strategy_analysis(games_df, embedding_model, use_pca=False):
    """
    Process each game from the perspective of each player:
      1) Build columns 'embedding_my'/'embedding_opponent'
      2) Run quantitative_analysis (distance-based) -> columns of lists
      3) Run qualitative_analysis (boolean WordNet-based) -> columns of lists
      4) decide_winning_strategy -> final per-round label
    """
    emb_col1 = f"embedding1_{embedding_model}"
    emb_col2 = f"embedding2_{embedding_model}"
    if use_pca:
        emb_col1 += "_pca"
        emb_col2 += "_pca"

    players = pd.concat([games_df['player1Id'], games_df['player2Id']]).unique()

    results = []

    for player in players:
        try:
            player_games = games_df[
                (games_df['player1Id'] == player) | (games_df['player2Id'] == player)
            ].copy()

            player_games['playerId'] = player

            # Build the "my" vs "opponent" embeddings & words
            player_games['embedding_my'] = player_games.apply(
                lambda row: np.array(row[emb_col1], dtype=float)
                            if row['player1Id'] == player else np.array(row[emb_col2], dtype=float),
                axis=1
            )
            player_games['embedding_opponent'] = player_games.apply(
                lambda row: np.array(row[emb_col2], dtype=float)
                            if row['player1Id'] == player else np.array(row[emb_col1], dtype=float),
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

            # 1) Quantitative
            player_games = quantitative_analysis(player_games)

            # 2) Qualitative
            player_games = qualitative_analysis(player_games)

            # 3) Decide winning strategy
            win_strat = player_games.apply(decide_winning_strategy, axis=1)
            player_games['winning_strategy_name'] = win_strat.apply(lambda x: x[0])
            player_games['winning_strategy_encoding'] = win_strat.apply(lambda x: x[1])

            results.append(player_games)

        except Exception as e:
            print(f"Error processing player {player}: {e}")

    return pd.concat(results, ignore_index=True)


##################################
#  CHOOSING THE 'WINNING' STRAT  #
##################################

def decide_winning_strategy(row):
    """
    For each round in the game row, decide which strategy 'wins'.
    Priority:
      1) If any of (synonym, abstraction, contrast) is 1 => choose among them in order: synonym > abstraction > contrast > morphological_variation
      2) Otherwise pick the min among the distance-based strategies:
         mirroring, balancing, staying_close, phonetic_distance, conceptual_expansion_distance
         (Be mindful that conceptual_expansion_distance is actually a *similarity*, so if you want
          to interpret "lowest distance" logic, you might do 1 - expansion_distance or handle it separately).
    """
    mirroring = row['mirroring_distance']
    balancing = row['balancing_distance']
    staying_close = row['staying_close_distance']
    expansion = row['conceptual_expansion_distance']  # this is a similarity

    abstraction = row['abstraction_measure']
    contrast = row['contrast_measure']
    synonym = row['synonym_measure']
    morph_variation = row['morphological_variation_measure']


    num_rounds = len(mirroring)  # they should all have the same length

    # We'll define a fixed order for encoding
    strategy_order = [
        "mirroring_distance",
        "balancing_distance",
        "staying_close_distance",
        "conceptual_expansion_distance",
        "abstraction_measure",
        "contrast_measure",
        "synonym_measure",
        "morphological_variation_measure"
    ]

    winning_names = []
    winning_encodings = []

    for i in range(num_rounds):
        if i == 0:
            winning_names.append(None)
            winning_encodings.append([np.nan]*len(strategy_order))
        else:
            bool_syn = synonym[i]
            bool_abs = abstraction[i]
            bool_con = contrast[i]
            bool_morph = morph_variation[i]

            if bool_syn == 1 or bool_abs == 1 or bool_con == 1 or bool_morph == 1:
                if bool_morph == 1:
                    chosen = "morphological_variation_measure"
                elif bool_syn == 1:
                    chosen = "synonym_measure"
                elif bool_abs == 1:
                    chosen = "abstraction_measure"
                elif bool_con == 1:
                    chosen = "contrast_measure"

            else:
                d_mirroring = mirroring[i]
                d_balancing = balancing[i]
                d_staying_close = staying_close[i]
                sim_expansion = expansion[i]
                d_expansion = 1 - sim_expansion if not np.isnan(sim_expansion) else float('inf')

                distances = {
                    "mirroring_distance": d_mirroring,
                    "balancing_distance": d_balancing,
                    "staying_close_distance": d_staying_close,
                    "conceptual_expansion_distance": d_expansion
                }
                chosen = min(distances, key=distances.get)

            # Record
            winning_names.append(chosen)
            encoding = [1 if s == chosen else 0 for s in strategy_order]
            winning_encodings.append(encoding)

    return (winning_names, winning_encodings)


###################################
#    HEATMAP + PRINTING RESULTS   #
###################################

def plot_strategy_heatmap(results_df, groupby='player'):
    """
    Plot a heatmap showing the average winning strategy usage,
    based on winning_strategy_encoding, which is a list of binary vectors per row.

    Strategy order (internal):
      [
         "mirroring_distance",
         "balancing_distance",
         "staying_close_distance",
         "conceptual_expansion_distance",
         "abstraction_measure",
         "contrast_measure",
         "synonym_measure"
         "morphological_variation_measure"
      ]
    """
    display_mapping = {
        "mirroring_distance": "mirroring",
        "balancing_distance": "balancing",
        "staying_close_distance": "staying_close",
        "conceptual_expansion_distance": "conceptual_expansion",
        "abstraction_measure": "abstraction",
        "contrast_measure": "contrast",
        "synonym_measure": "synonym",
        "morphological_variation_measure": "morphological_variation"
    }

    strategy_order = list(display_mapping.keys())

    rows = []
    for idx, row in results_df.iterrows():
        if groupby == 'player':
            group_val = row.get("playerId", "Unknown")
        elif groupby == 'game':
            if "botId" in row:
                if pd.isna(row["botId"]) or row["botId"] == "":
                    group_val = "Human vs Human"
                else:
                    group_val = "Human vs Bot"
            else:
                group_val = "Unknown"
        else:
            raise ValueError("groupby must be either 'player' or 'game'")

        encoding_list = row.get("winning_strategy_encoding", None)
        if encoding_list is not None and isinstance(encoding_list, list):
            arr = np.array(encoding_list, dtype=float)  # shape (num_rounds, 8)
            avg_vals = np.nanmean(arr, axis=0)  # mean across rounds
        else:
            avg_vals = np.full(len(strategy_order), np.nan)

        for i, strategy in enumerate(strategy_order):
            rows.append({
                "group": group_val,
                "strategy": strategy,
                "value": avg_vals[i]
            })

    long_df = pd.DataFrame(rows)
    grouped = long_df.groupby(["group", "strategy"])["value"].mean().reset_index()
    pivoted = grouped.pivot(index="group", columns="strategy", values="value")
    pivoted.rename(columns=display_mapping, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(pivoted.values, cmap="coolwarm", aspect="auto", interpolation="nearest")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax.set_xticks(np.arange(len(pivoted.columns)))
    ax.set_yticks(np.arange(len(pivoted.index)))
    ax.set_xticklabels(pivoted.columns, rotation=90)
    ax.set_yticklabels(pivoted.index)

    if groupby == 'player':
        ax.set_ylabel("Player ID")
        ax.set_title("Average Winning Strategy Measures by Player (Percentage)")
    else:
        ax.set_ylabel("Game Configuration")
        ax.set_title("Average Winning Strategy Measures by Game Configuration (Percentage)")

    ax.set_xlabel("Strategy")

    # Annotate cells with percentage
    for i in range(pivoted.shape[0]):
        for j in range(pivoted.shape[1]):
            val = pivoted.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val * 100:.0f}%", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

def print_game_turns(results_df, n=20, filter_strategies=None):
    """
    Print turn details for the first n rows (games) in results_df.
    If filter_strategies is provided, only print turns whose winning strategy is in that set.
    """
    display_mapping = {
        "mirroring_distance": "mirroring",
        "balancing_distance": "balancing",
        "staying_close_distance": "staying_close",
        "conceptual_expansion_distance": "conceptual_expansion",
        "abstraction_measure": "abstraction",
        "contrast_measure": "contrast",
        "synonym_measure": "synonym",
        "morphological_variation_measure": "morphological_variation"
    }

    for idx, row in results_df.head(n).iterrows():
        print(f"Game {idx}:")
        word_my = eval(row['word_my'])
        word_opponent = eval(row['word_opponent'])
        winning_strats = row['winning_strategy_name']
        rounds = min(len(word_my), len(word_opponent), len(winning_strats))

        if rounds < 2:
            print("  Not enough rounds to display details.\n")
            continue

        for i in range(1, rounds):
            prev_pword = word_my[i-1]
            prev_oword = word_opponent[i-1]
            curr_pword = word_my[i]
            strat = winning_strats[i]

            if strat is None:
                continue
            if filter_strategies and strat not in filter_strategies:
                continue

            readable_strat = display_mapping.get(strat, strat)
            print(f"  Turn {i}: [{prev_pword} / {prev_oword}] -> {curr_pword}  (winning strategy: {readable_strat})")
        print()

if __name__ == "__main__":
    import os
    from scipy.spatial.distance import cosine
    from utils.embeding_utils import (get_embeddings_for_table, calculate_pca_for_embeddings,
                                     plot_embedding_distance_during_game,
                                     plot_distance_evolution_per_player)
    from game_statistics import calculate_game_metrics_per_configuration

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
    player_metrics = calculate_game_metrics_per_configuration(games_df)
    print("Success Rate and Average Rounds for Winning Games:")
    print(player_metrics)

    # 5) Strategy analysis (using the PCA columns):
    results_df = strategy_analysis(games_df, embedding_model, use_pca=True)
    plot_strategy_heatmap(results_df, groupby='game')
    # plot_strategy_heatmap(results_df)

    # 4) Plot distances with the original or PCA embeddings
    plot_embedding_distance_during_game(
        results_df,
        distance_func=cosine,
        embedding_model="glove",
        use_pca=True,
        align_end=True,
    )
    plot_embedding_distance_during_game(
        results_df,
        distance_func=cosine,
        embedding_model="glove",
        use_pca=False,
        align_end=True,
    )
    plot_distance_evolution_per_player(
        results_df,
        distance_func=cosine,
        embedding_model="glove",
        use_pca=True,
        last_rounds=5,
    )

    print_game_turns(results_df, n=5)
