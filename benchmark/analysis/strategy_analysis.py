import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# External module imports
from benchmark.analysis.utils.data_loading import load_sql_data
from quantitative_analysis import min_max_normalize, assign_quantitative_strategy, quantitative_analysis
from qualitative_analysis import assign_qualitative_strategy, qualitative_analysis

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
            "conceptual_linking",
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
    from game_statistics import calculate_game_metrics_per_configuration, calculate_game_metrics_per_player

    db_name = "downloaded_word_sync_20250210_195900.db"
    csv_name = "games.csv"

    # 1) Load the data
    if not os.path.exists(csv_name):
        players_df, games_df = load_sql_data(db_name)
        games_df.to_csv(csv_name, index=False)
    else:
        games_df = pd.read_csv(csv_name)

    # 2) Get embeddings (and do PCA with e.g. 50 components)
    embedding_model = "word2vec"
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

    # 4) Calculate player metrics
    player_metrics = calculate_game_metrics_per_player(games_df)
    print("Average Number of Rounds and Success Rate per Player:")
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

    print_game_turns(results_df, n=5, )
