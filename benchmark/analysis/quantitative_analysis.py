import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


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
        if isinstance(game['word_my'], list):
            word_my = game['word_my']
            word_opponent = game['word_opponent']
        else:
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
