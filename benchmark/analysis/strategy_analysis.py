import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# External module imports
from benchmark.analysis.quantitative_analysis import assign_quantitative_strategy, quantitative_analysis
from benchmark.analysis.qualitative_analysis.qualitative_analysis import assign_semantic_strategy, qualitative_analysis

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
        # Check the output of the apply method
        player_games['semantic_strategy_name'] = None
        player_games['quantitative_strategy_name'] = None
        player_games.loc[:, 'semantic_strategy_name'] = player_games.apply(assign_semantic_strategy, axis=1)
        player_games.loc[:, 'quantitative_strategy_name'] = player_games.apply(assign_quantitative_strategy, axis=1)

        results.append(player_games)

    return pd.concat(results, ignore_index=True)

def plot_strategy_heatmap(results_df, strategy_col="semantic_strategy_name", groupby='player'):
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
        e.g. "semantic_strategy_name" or "quantitative_strategy_name".
    groupby : {'player', 'configuration'}
        How to group the data in the heatmap.
    """

    # 1) Define the set of possible labels for your chosen strategy_col.
    #    Adjust as needed to match your actual labels.
    if strategy_col == "semantic_strategy_name":
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
        elif groupby == 'configuration':
            # Example: Distinguish 'Human vs Bot' vs. 'Human vs Human'
            if "botId" in row:
                if pd.isna(row["botId"]) or row["botId"] == "":
                    group_val = "Human vs Human"
                else:
                    group_val = "Human vs Bot"
            else:
                group_val = "Unknown"
        else:
            raise ValueError("groupby must be either 'player' or 'configuration'.")

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

def print_game_turns(results_df, n=5):
    """
    Example function that prints both words and
    multi-label qualitative strategy for each round.
    """

    for idx, row in results_df.head(n).iterrows():
        print(f"Game {idx}:")
        word_my = row["word_my"]
        word_opponent = row["word_opponent"]

        q_strats = row.get("semantic_strategy_name", [])
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

def print_scores(results_df):
    for idx, row in results_df.iterrows():
        if pd.isna(row["botId"]) or row["botId"] == "":
            group_val = "Human vs Human"
        else:
            group_val = "Human vs Bot"

        # add the group_val to the row
        results_df.at[idx, "configuration"] = group_val
        # for each row, calculate conceptual_linking_score_avg and collocation_score_avg
        results_df.at[idx, "conceptual_linking_score_avg"] = np.nanmean(row["conceptual_linking_score"])
        results_df.at[idx, "collocation_score_avg"] = np.nanmean(row["collocation_score"])

    # Group by configuration and compute the average scores
    avg_scores = results_df.groupby("configuration").agg({
        "conceptual_linking_score_avg": "mean",
        "collocation_score_avg": "mean"
    }).reset_index()

    print(avg_scores)


if __name__ == "__main__":
    import os
    from utils.embeding_utils import (get_embeddings_for_table, calculate_pca_for_embeddings)
    from game_statistics import calculate_game_metrics_per_player
    from benchmark.analysis.utils.data_loading import load_sql_data, load_csv

    db_name = "downloaded_word_sync_20250210_195900.db"
    csv_name = "games.csv"

    # 1) Load the data
    player_ids = ["l3o0u7Bo", "bNgvBUv7", "koLvZAXK", "pZrgT64W"]
    if not os.path.exists(csv_name):
        players_df, games_df = load_sql_data(db_name, player_ids=player_ids)
        games_df.to_csv(csv_name, index=False)
    else:
        games_df = pd.read_csv(csv_name)

    # 2) Get embeddings (and do PCA with e.g. 50 components)
    embedding_model = "word2vec"
    games_df = get_embeddings_for_table( games_df, model_name=embedding_model,)
    games_df = calculate_pca_for_embeddings(
        games_df,
        model_name=embedding_model,
        num_pca_components=15,
    )

    # Save to CSV for future use
    games_df.to_csv(csv_name, index=False)

    # 3) Calculate player metrics
    # player_metrics = calculate_game_metrics_per_configuration(games_df, plot_box=True)
    print("Success Rate and Average Rounds for Winning Games:")
    # print(player_metrics)

    # 4) Calculate player metrics
    player_metrics = calculate_game_metrics_per_player(games_df)
    print("Average Number of Rounds and Success Rate per Player:")
    print(player_metrics)


    # 5) Strategy analysis
    strategy_results_file = "results.csv"
    if not os.path.exists(strategy_results_file):
        results_df = strategy_analysis(games_df, embedding_model, use_pca=True)
        # Save results with strategies to CSV but only a subset of columns
        cols_to_save = [
            "gameId",
            "playerId",
            "player1Id",
            "player2Id",
            "botId",
            "status",
            "roundCount",
            "wordsPlayed1",
            "wordsPlayed2",
            "word_my",
            "word_opponent",
            "surveyAnswers1",
            "surveyAnswers2",
            "semantic_strategy_name",
            "quantitative_strategy_name",
            "conceptual_linking_score",
            "collocation_score",
        ]
        results_df_partial = results_df[cols_to_save]
        results_df_partial.to_csv("results.csv", index=False)
    else:
        results_df = load_csv(strategy_results_file, columns_to_convert=["wordsPlayed1", "wordsPlayed2", "semantic_strategy_name", "quantitative_strategy_name", "conceptual_linking_score", "collocation_score"])

    # 6) Group by configuration and compute the average scores
    print_scores(results_df)

    plot_strategy_heatmap(results_df, strategy_col="semantic_strategy_name", groupby='configuration')
    plot_strategy_heatmap(results_df, strategy_col="quantitative_strategy_name", groupby='configuration')
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
