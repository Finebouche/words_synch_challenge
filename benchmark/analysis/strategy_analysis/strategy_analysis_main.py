import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# External module imports
from .quantitative_analysis import assign_quantitative_strategy, quantitative_analysis
from .qualitative_analysis import assign_semantic_strategy, qualitative_analysis, colloquial_conceptual_linking_analysis

def strategy_analysis(games_df, embedding_model, use_pca=False, use_conceptual_linking_score=True):
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

    for player in tqdm(players, desc="Processing players"):
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

        # Scores for conceptual linking and collocation
        player_games = colloquial_conceptual_linking_analysis(player_games, use_conceptual_linking_score=use_conceptual_linking_score)

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
    in `strategy_col`, grouped by either 'player', 'configuration', or 'gameConfig'.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing at least:
            - 'playerId' (for grouping by 'player' or 'gameConfig'),
            - 'player1Id' and 'player2Id' columns,
            - 'gameConfigPlayer1' and 'gameConfigPlayer2' columns,
            - a column `strategy_col` that is a list of lists of labels,
              one sub-list per round (for multi-strategy ties),
            - and other columns as needed (e.g., 'botId', 'word_my', 'word_opponent').
    strategy_col : str
        The DataFrame column name that holds a list of *lists* of strategy labels,
        e.g. "semantic_strategy_name" or "quantitative_strategy_name".
    groupby : {'player', 'configuration', 'gameConfig'}, default='player'
        How to group the data in the heatmap.
          - 'player': groups by the "playerId" column.
          - 'configuration': groups into "Human vs Human" vs. "Human vs Bot" based on botId.
          - 'gameConfig': groups by the game configuration for the player.
             For each row, if playerId == player1Id then use gameConfigPlayer1;
             if playerId == player2Id then use gameConfigPlayer2.
    """
    # 1) Define the set of possible labels and a mapping for prettier names.
    if strategy_col == "semantic_strategy_name":
        possible_strategies = [
            "none", "synonym", "morphological_variation", "abstraction",
            "contrast", "thematic_alignment", "meronym", "troponymy"
        ]
        strategy_label_map = {
            "none": "None",
            "synonym": "Synonym",
            "morphological_variation": "Morph. Var.",
            "abstraction": "Abstraction",
            "contrast": "Contrast",
            "thematic_alignment": "Thematic Align.",
            "meronym": "Meronym",
            "troponymy": "Troponymy"
        }
    elif strategy_col == "quantitative_strategy_name":
        possible_strategies = ["mirroring", "balancing", "staying_close"]
        strategy_label_map = {
            "mirroring": "Mirroring",
            "balancing": "Balancing",
            "staying_close": "Staying\nClose"
        }
    else:
        raise ValueError(f"Unknown strategy column: {strategy_col}")

    # 2) Accumulate rows in "long" format.
    rows = []
    for idx, row in results_df.iterrows():
        # Determine the group value based on the chosen grouping method.
        if groupby == 'player':
            group_val = row.get("playerId", "Unknown")
        elif groupby in ['configuration', 'gameConfig']:
            # For both 'configuration' and 'gameConfig', try to use player's config.
            player = row.get("playerId", None)
            if player is not None:
                if player == row.get("player1Id"):
                    group_val = row.get("gameConfigPlayer1", "Unknown")
                elif player == row.get("player2Id"):
                    group_val = row.get("gameConfigPlayer2", "Unknown")
                else:
                    group_val = "Unknown"
            else:
                group_val = "Unknown"
        else:
            raise ValueError("groupby must be one of: 'player', 'configuration', or 'gameConfig'.")

        # Retrieve the list-of-lists for strategies.
        strategy_list = row.get(strategy_col, None)
        freq_dict = {s: 0 for s in possible_strategies}
        total_count = 0

        if isinstance(strategy_list, list) and len(strategy_list) > 0:
            for round_labels in strategy_list:
                if not round_labels:
                    continue
                if not isinstance(round_labels, list):
                    round_labels = [round_labels]
                total_count += len(round_labels)
                for lab in round_labels:
                    if lab in freq_dict:
                        freq_dict[lab] += 1
                    else:
                        freq_dict["none"] += 1
        else:
            total_count = 0

        if total_count > 0:
            for s in possible_strategies:
                freq_dict[s] = freq_dict[s] / total_count
        else:
            for s in possible_strategies:
                freq_dict[s] = np.nan

        for s in possible_strategies:
            rows.append({
                "group": group_val,
                "strategy": s,
                "value": freq_dict[s]
            })

    # 3) Create a DataFrame and pivot it.
    long_df = pd.DataFrame(rows)
    grouped_df = long_df.groupby(["group", "strategy"])["value"].mean().reset_index()
    pivoted = grouped_df.pivot(index="group", columns="strategy", values="value")

    # 4) Map game configuration names if grouping is 'configuration' or 'gameConfig'.
    if groupby in ['configuration', 'gameConfig']:
        config_map = {
            "human_vs_human_(human_shown)": "H-H\n(Human shown)",
            "human_vs_human_(bot_shown)": "H-H\n(AI shown)",
            "human_vs_bot_(bot_shown)": "H-AI\n(AI shown)",
            "human_vs_bot_(human_shown)": "H-AI\n(Human shown)"
        }
        pivoted.index = pivoted.index.map(lambda x: config_map.get(x, x))

    # 5) Update x-axis labels using the strategy mapping.
    new_strategy_labels = [strategy_label_map.get(s, s) for s in pivoted.columns]

    # 6) Plot the heatmap.
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(
        pivoted.values,
        cmap="coolwarm",
        aspect="auto",
        interpolation="nearest",
        vmin=0.17,
        vmax=0.51
    )

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Set x-axis and y-axis tick labels.
    ax.set_xticks(np.arange(len(pivoted.columns)))
    ax.set_yticks(np.arange(len(pivoted.index)))
    ax.set_xticklabels(new_strategy_labels, rotation=0)
    ax.set_yticklabels(pivoted.index)

    # Force the x-axis ticks and label to appear at the bottom.
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")

    # Set appropriate axis labels and title.
    if groupby == 'player':
        ax.set_ylabel("Player ID")
        title_grouping = "by Player"
    elif groupby == 'configuration':
        ax.set_ylabel("Game Configuration")
        title_grouping = "per Game Configuration"
    elif groupby == 'gameConfig':
        title_grouping = "per Game Configuration"
    else:
        title_grouping = ""

    ax.set_title(f"Average Strategy Usage Frequency {title_grouping} (as %)")

    # Annotate each cell with percentage values.
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
