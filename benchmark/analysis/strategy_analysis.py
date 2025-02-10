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
# QUALITATIVE ANALYSIS  #
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


##############################
# NEW LEXICAL RELATIONS      #
##############################

def is_meronym(candidate_word, original_word):
    """
    Return True if candidate_word is a meronym of original_word.
    Checks if any synset of candidate_word appears in any of the meronym lists
    (part, substance, or member meronyms) of any synset of original_word.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    for o_syn in original_synsets:
        for meronym_func in [lambda s: s.part_meronyms(),
                             lambda s: s.substance_meronyms(),
                             lambda s: s.member_meronyms()]:
            meronyms = meronym_func(o_syn)
            if any(c_syn in meronyms for c_syn in candidate_synsets):
                return True
    return False


def is_holonym(candidate_word, original_word):
    """
    Return True if candidate_word is a holonym of original_word.
    Checks if any synset of candidate_word appears in any of the holonym lists
    (part, substance, or member holonyms) of any synset of original_word.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    for o_syn in original_synsets:
        for holonym_func in [lambda s: s.part_holonyms(),
                             lambda s: s.substance_holonyms(),
                             lambda s: s.member_holonyms()]:
            holonyms = holonym_func(o_syn)
            if any(c_syn in holonyms for c_syn in candidate_synsets):
                return True
    return False


def is_troponym(candidate_word, original_word):
    """
    Return True if candidate_word is a troponym of original_word.
    (Troponyms capture the manner of performing a verb.)
    Applicable for verbs.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word, pos=wn.VERB)
    original_synsets = wn.synsets(original_word, pos=wn.VERB)

    for o_syn in original_synsets:
        # Use try/except to handle if 'troponyms' is not available
        try:
            troponyms = o_syn.troponyms()
        except AttributeError:
            troponyms = []  # Fallback if not implemented
        if any(c_syn in troponyms for c_syn in candidate_synsets):
            return True
    return False


def is_entailment(candidate_word, original_word):
    """
    Return True if candidate_word is entailed by original_word.
    (Entailment in verbs indicates that the occurrence of one action implies another.)
    Applicable for verbs.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word, pos=wn.VERB)
    original_synsets = wn.synsets(original_word, pos=wn.VERB)

    for o_syn in original_synsets:
        entailments = o_syn.entailments()
        if any(c_syn in entailments for c_syn in candidate_synsets):
            return True
    return False

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
    Compute multiple boolean-based measures per round, including:
      - abstraction_measure: is current_word a hypernym of opponent's prev word OR hyponym of player's own prev word?
      - contrast_measure: is current_word an antonym of opponent's prev word OR player's own prev word?
      - synonym_measure: is current_word a synonym of opponent's prev word OR player's own prev word?
      - morphological_variation_measure: are current_word and previous words morphological variations?
      - thematic_alignment_measure: do current_word and previous words share a broad category?
      - meronym_measure: is current_word a meronym of opponent's prev word OR a holonym of player's own prev word?
      - troponymy_measure: (for verbs) is current_word a troponym of opponent's prev word OR entailed by player's own prev word?

    Each measure is stored as a list (with length = number of rounds) containing 0/1 values
    (or np.nan for round 0).
    """
    # Initialize new columns
    player_games['abstraction_measure'] = None
    player_games['contrast_measure'] = None
    player_games['synonym_measure'] = None
    player_games['morphological_variation_measure'] = None
    player_games['thematic_alignment_measure'] = None
    player_games['meronym_measure'] = None
    player_games['troponymy_measure'] = None

    for index, game in player_games.iterrows():
        # Convert string representations to lists if necessary
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])
        num_rounds = min(len(word_my), len(word_opponent))

        abstraction_list = []
        contrast_list = []
        synonym_list = []
        morph_variation_list = []
        thematic_alignment_list = []
        meronym_list = []
        troponymy_list = []

        for i in range(num_rounds):
            if i == 0:
                abstraction_list.append(np.nan)
                contrast_list.append(np.nan)
                synonym_list.append(np.nan)
                morph_variation_list.append(np.nan)
                thematic_alignment_list.append(np.nan)
                meronym_list.append(np.nan)
                troponymy_list.append(np.nan)
            else:
                current_word = word_my[i]
                prev_opponent_word = word_opponent[i - 1]
                prev_my_word = word_my[i - 1]

                abstraction_score = int(is_hypernym(current_word, prev_opponent_word)) + int(
                    is_hyponym(current_word, prev_my_word))
                contrast_score = int(is_antonym(current_word, prev_opponent_word)) + int(
                    is_antonym(current_word, prev_my_word))
                synonym_score = int(is_synonym(current_word, prev_opponent_word)) + int(
                    is_synonym(current_word, prev_my_word))
                morph_variation_score = int(is_morphological_variation(current_word, prev_opponent_word)) + int(
                    is_morphological_variation(current_word, prev_my_word))
                thematic_alignment_score = int(is_thematic_alignment(current_word, prev_opponent_word)) + int(
                    is_thematic_alignment(current_word, prev_my_word))
                meronym_score = int(is_meronym(current_word, prev_opponent_word)) + int(
                    is_holonym(current_word, prev_my_word))
                troponymy_score = int(is_troponym(current_word, prev_opponent_word)) + int(
                    is_entailment(current_word, prev_my_word))

                abstraction_list.append(abstraction_score)
                contrast_list.append(contrast_score)
                synonym_list.append(synonym_score)
                morph_variation_list.append(morph_variation_score)
                thematic_alignment_list.append(thematic_alignment_score)
                meronym_list.append(meronym_score)
                troponymy_list.append(troponymy_score)

        player_games.at[index, 'abstraction_measure'] = abstraction_list
        player_games.at[index, 'contrast_measure'] = contrast_list
        player_games.at[index, 'synonym_measure'] = synonym_list
        player_games.at[index, 'morphological_variation_measure'] = morph_variation_list
        player_games.at[index, 'thematic_alignment_measure'] = thematic_alignment_list
        player_games.at[index, 'meronym_measure'] = meronym_list
        player_games.at[index, 'troponymy_measure'] = troponymy_list

    return player_games

##############################
# QUANTITATIVE (DISTANCES)  #
##############################

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

    for index, game in player_games.iterrows():
        embedding_my = game['embedding_my']
        embedding_opponent = game['embedding_opponent']
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])

        num_rounds = min(len(embedding_my), len(embedding_opponent), len(word_my), len(word_opponent))

        mirroring_list = []
        balancing_list = []
        staying_close_list = []

        for i in range(num_rounds):
            if i == 0:
                mirroring_list.append(np.nan)
                balancing_list.append(np.nan)
                staying_close_list.append(np.nan)
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

        # Store raw results:
        player_games.at[index, 'mirroring_distance'] = mirroring_list
        player_games.at[index, 'balancing_distance'] = balancing_list
        player_games.at[index, 'staying_close_distance'] = staying_close_list

        # Normalize each measure for this game (per row normalization across rounds)
        norm_mirroring = min_max_normalize(mirroring_list)
        norm_balancing = min_max_normalize(balancing_list)
        norm_staying_close = min_max_normalize(staying_close_list)

        player_games.at[index, 'mirroring_distance'] = norm_mirroring
        player_games.at[index, 'balancing_distance'] = norm_balancing
        player_games.at[index, 'staying_close_distance'] = norm_staying_close

    return player_games

##################################
#  MAIN STRATEGY ANALYSIS LOOP   #
##################################

def assign_qualitative_strategy(row):
    """
    For each round in the game, assign one OR MORE labels based on the boolean-like
    WordNet measures. We now accept multiple 'winning' strategies if they tie
    for the highest numeric score in that round.

    Measures (and their meaning):
        - abstraction_measure         (hypernym/hyponym checks)
        - contrast_measure            (antonym checks)
        - synonym_measure             (synonym checks)
        - morphological_variation_measure
        - thematic_alignment_measure
        - meronym_measure
        - troponymy_measure

    Behavior:
        - If all measures are zero, store ["none"].
        - Otherwise, find the maximum integer score among these measures
        (e.g., 2) and collect ALL measure names that match that max score (> 0).
        - Append that list for each round, so we end up with a list of lists:
          [ ["none"], ["synonym", "contrast"], ["abstraction"], ... ]
    """
    # Retrieve the lists from the row (each is a list of per-round integers or np.nan)
    abstraction = row['abstraction_measure']
    contrast = row['contrast_measure']
    synonym = row['synonym_measure']
    morph = row['morphological_variation_measure']
    thematic = row['thematic_alignment_measure']
    meronym = row['meronym_measure']
    troponymy = row['troponymy_measure']

    num_rounds = len(abstraction)  # all should have the same length
    strategy_labels = []

    for i in range(num_rounds):
        # If round i is NaN (often round 0), just store None
        if pd.isna(abstraction[i]):
            strategy_labels.append(None)
            continue

        # Build a dict: measure_name -> measure_value
        measure_values = {
            'synonym': synonym[i],
            'morphological_variation': morph[i],
            'abstraction': abstraction[i],
            'contrast': contrast[i],
            'thematic_alignment': thematic[i],
            'meronym': meronym[i],
            'troponymy': troponymy[i],
        }

        # If all measures == 0, store ["none"]
        if all(val == 0 for val in measure_values.values()):
            winning_strats = ["none"]
        else:
            # Determine the maximum integer value
            max_value = max(measure_values.values())

            # Collect all measure names that match max_value (and > 0)
            winning_strats = [
                mname
                for mname, mval in measure_values.items()
                if mval == max_value and mval > 0
            ]

            # If the max_value is 0, or for some reason no measures qualified, fallback to ["none"]
            if not winning_strats:
                winning_strats = ["none"]

        # Append the list of winning strategies (could be 1 or multiple)
        strategy_labels.append(winning_strats)

    return strategy_labels

def assign_quantitative_strategy(row):
    """
    For each round, pick the label corresponding to the *lowest distance* among
    the quantitative distance measures.

    Measures:
     - mirroring_distance
     - balancing_distance
     - staying_close_distance
    """
    mirroring = row['mirroring_distance']
    balancing = row['balancing_distance']
    staying_close = row['staying_close_distance']

    num_rounds = len(mirroring)  # they should have the same length
    strategy_labels = []

    for i in range(num_rounds):
        if pd.isna(mirroring[i]):
            strategy_labels.append(None)
            continue

        # We pick the measure with minimum distance.
        distances = {
            'mirroring': mirroring[i],
            'balancing': balancing[i],
            'staying_close': staying_close[i]
        }

        # If all are valid floats, find the minimum
        chosen = min(distances, key=distances.get)
        strategy_labels.append(chosen)

    return strategy_labels


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

            # 3) Decide winning strategies
            # Apply to each row
            player_games['qualitative_strategy_name'] = player_games.apply(assign_qualitative_strategy, axis=1)
            player_games['quantitative_strategy_name'] = player_games.apply(assign_quantitative_strategy, axis=1)

            results.append(player_games)

        except Exception as e:
            print(f"Error processing player {player}: {e}")

    return pd.concat(results, ignore_index=True)


def plot_strategy_heatmap(
    results_df,
    strategy_col="qualitative_strategy_name",
    groupby='player'
):
    """
    Plot a heatmap showing average usage frequency of each strategy label
    in `strategy_col`, grouped by either 'player' or 'game'.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing at least:
            - 'playerId' or 'botId' columns (for grouping),
            - a column `strategy_col` that is a list of lists of labels,
              one sub-list per round (for multi-strategy ties),
            - 'word_my'/'word_opponent' (or at least consistent # of rounds).
    strategy_col : str
        The DataFrame column name that holds a list of *lists* of strategy labels,
        e.g. "qualitative_strategy_name" or "quantitative_strategy_name".
    groupby : {'player', 'game'}
        How to group the data in the heatmap.
    """

    # 1) Define the set of possible labels for your chosen strategy_col.
    #    Adjust as needed to match your actual labels.
    if strategy_col == "qualitative_strategy_name":
        possible_strategies = [
            "none",
            "synonym",
            "morphological_variation",
            "abstraction",
            "contrast",
            "thematic_alignment",
            "meronym",
            "troponymy",
        ]
    elif strategy_col == "quantitative_strategy_name":
        possible_strategies = [
            "mirroring",
            "balancing",
            "staying_close",
        ]
    else:
        # Fallback or a custom list. E.g., you could parse the unique labels:
        # possible_strategies = find_unique_strategies(results_df[strategy_col])
        raise ValueError(f"Unknown strategy column: {strategy_col}")

    # 2) Prepare to accumulate rows => we'll build a "long format" table.
    rows = []

    # 3) For each row (one game from one player's perspective),
    #    compute the frequency of each label across the rounds.
    for idx, row in results_df.iterrows():
        # Determine the group value for this row
        if groupby == 'player':
            group_val = row.get("playerId", "Unknown")
        elif groupby == 'game':
            # Example: Distinguish 'Human vs Bot' vs. 'Human vs Human'
            if "botId" in row:
                if pd.isna(row["botId"]) or row["botId"] == "":
                    group_val = "Human vs Human"
                else:
                    group_val = "Human vs Bot"
            else:
                group_val = "Unknown"
        else:
            raise ValueError("groupby must be either 'player' or 'game'.")

        # Get the "strategy_list", which should be a list of lists,
        # e.g. [ ["synonym"], ["synonym","contrast"], ["none"], ... ]
        strategy_list = row.get(strategy_col, None)

        # Initialize a frequency dict for the possible strategies
        freq_dict = {s: 0 for s in possible_strategies}
        total_count = 0  # total number of label occurrences across all rounds

        if isinstance(strategy_list, list) and len(strategy_list) > 0:
            # For each round's sub-list of labels
            for round_labels in strategy_list:
                # If it's None, empty, or not a list, handle accordingly
                if not round_labels:
                    # e.g., None or []
                    continue
                if not isinstance(round_labels, list):
                    # If for some reason it's a single string, wrap in a list
                    round_labels = [round_labels]

                # Count how many labels appear this round
                total_count += len(round_labels)
                for lab in round_labels:
                    # If label is recognized, increment it; otherwise, increment "none"
                    if lab in freq_dict:
                        freq_dict[lab] += 1
                    else:
                        freq_dict["none"] += 1
        else:
            # If no data for this row, mark everything as NaN
            total_count = 0

        # Convert raw counts to frequencies
        if total_count > 0:
            for s in possible_strategies:
                freq_dict[s] = freq_dict[s] / total_count
        else:
            for s in possible_strategies:
                freq_dict[s] = np.nan

        # Collect rows for "long format"
        for s in possible_strategies:
            rows.append({
                "group": group_val,
                "strategy": s,
                "value": freq_dict[s]
            })

    # 4) Convert to DataFrame, then group by ("group","strategy"), compute mean, pivot
    long_df = pd.DataFrame(rows)
    grouped = long_df.groupby(["group", "strategy"])["value"].mean().reset_index()
    pivoted = grouped.pivot(index="group", columns="strategy", values="value")

    # 5) Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(
        pivoted.values,
        cmap="coolwarm",
        aspect="auto",
        interpolation="nearest",
        vmin=0,  # Frequencies from 0..1
        vmax=1,
    )

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax.set_xticks(np.arange(len(pivoted.columns)))
    ax.set_yticks(np.arange(len(pivoted.index)))
    ax.set_xticklabels(pivoted.columns, rotation=90)
    ax.set_yticklabels(pivoted.index)

    if groupby == 'player':
        ax.set_ylabel("Player ID")
        title_grouping = "by Player"
    else:
        ax.set_ylabel("Game Configuration")
        title_grouping = "by Game Configuration"

    ax.set_xlabel("Strategy")
    ax.set_title(f"Average '{strategy_col}' Usage Frequency {title_grouping} (as %)")

    # 6) Annotate cells with percentage
    for i in range(pivoted.shape[0]):
        for j in range(pivoted.shape[1]):
            val = pivoted.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val * 100:.0f}%", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

def print_game_turns(
        results_df,
        n=5
):
    """
    Example function that prints both words and
    multi-label qualitative strategy for each round.
    """

    for idx, row in results_df.head(n).iterrows():
        print(f"Game {idx}:")
        word_my = row["word_my"]
        word_opponent = row["word_opponent"]

        q_strats = row.get("qualitative_strategy_name", [])
        t_strats = row.get("quantitative_strategy_name", [])

        # If these are strings, parse them:
        if isinstance(word_my, str):
            word_my = eval(word_my)
        if isinstance(word_opponent, str):
            word_opponent = eval(word_opponent)

        num_rounds = min(len(word_my), len(word_opponent), len(q_strats), len(t_strats))
        if num_rounds < 2:
            print("  Not enough rounds.\n")
            continue

        for i in range(1, num_rounds):
            prev_pword = word_my[i - 1]
            prev_oword = word_opponent[i - 1]
            curr_pword = word_my[i]

            # Qual might be a list of strategies:
            q_label_list = q_strats[i] if i < len(q_strats) else None
            # Convert e.g. ["synonym", "contrast"] -> "synonym,contrast"
            if isinstance(q_label_list, list):
                q_label_str = ",".join(q_label_list)
            else:
                q_label_str = str(q_label_list)

            # Quant normally a single label:
            t_label = t_strats[i] if i < len(t_strats) else None

            print(
                f"  Turn {i}: [{prev_pword} / {prev_oword}] -> {curr_pword}  "
                f"(qualitative: {q_label_str}, quantitative: {t_label})"
            )

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
    plot_strategy_heatmap(results_df, strategy_col="qualitative_strategy_name", groupby='game')
    plot_strategy_heatmap(results_df, strategy_col="quantitative_strategy_name", groupby='game')
    # plot_strategy_heatmap(results_df)

    # 4) Plot distances with the original or PCA embeddings
    # plot_embedding_distance_during_game(
    #     results_df,
    #     distance_func=cosine,
    #     embedding_model="glove",
    #     use_pca=True,
    #     align_end=True,
    # )
    # plot_distance_evolution_per_player(
    #     results_df,
    #     distance_func=cosine,
    #     embedding_model="glove",
    #     use_pca=True,
    #     last_rounds=5,
    # )

    print_game_turns(results_df, n=5)
