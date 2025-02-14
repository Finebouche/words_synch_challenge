import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(df):
    """Compute success rate, its standard deviation, average rounds, and round std for a given DataFrame."""
    success = (df['status'] == 'won').astype(float)
    return {
        'success_rate': success.mean(),
        'success_std': success.std(),
        'average_rounds': df['roundCount'].mean(),
        'rounds_std': df['roundCount'].std()
    }


def calculate_game_metrics_per_configuration(games_df, plot_box=False, separate_per_config=False):
    """
    Computes game metrics (success rate, average rounds, and their standard deviations)
    and returns them as a structured dictionary.

    There are two modes:
    1. If separate_per_config is False (default), games are grouped into:
         - Human-only: both players' configurations contain "vs_human"
         - Bot-involved: at least one player's configuration contains "vs_bot"
         - Total: all games.
    2. If separate_per_config is True, metrics are computed for each unique
       combination of 'gameConfigPlayer1' and 'gameConfigPlayer2'.

    Optionally, a box plot of the round count distributions is generated.

    Parameters:
        games_df (DataFrame): Data containing 'gameConfigPlayer1', 'gameConfigPlayer2',
                              'status', and 'roundCount'.
        plot_box (bool): If True, display a box plot of round counts.
        separate_per_config (bool): If True, compute metrics per unique configuration pair.

    Returns:
        dict: A dictionary with computed metrics.
              - If separate_per_config is False, keys are 'human', 'bot', and 'total'.
              - Otherwise, keys are the configuration combinations.
    """
    df = games_df.copy()

    #remove the outliers where  round count is 1 or 2
    df = df[df['roundCount'] >= 3]
    # if roundCount is over 16 we consider the game at loss and we set the roundCount to 16 and the status to loss
    df.loc[df['roundCount'] > 16, 'roundCount'] = 16
    df.loc[df['roundCount'] == 16, 'status'] = 'loss'


    if separate_per_config:
        # Create new columns for the player configuration and ID.
        # For all rows, set defaults using player1 values.
        df['gameConfigPlayer'] = df['gameConfigPlayer1']
        df['playerId'] = df['player1Id']

        # Create a duplicate of the rows where trueGameConfig is 'human_vs_human'.
        dup = df[df['trueGameConfig'] == 'human_vs_human'].copy()
        dup['gameConfigPlayer'] = dup['gameConfigPlayer2']
        dup['playerId'] = dup['player2Id']

        # Concatenate the original DataFrame with the duplicate.
        df = pd.concat([df, dup], ignore_index=True)

        grouped = df.groupby('gameConfigPlayer')

        # Compute metrics per configuration using dictionary comprehension.
        metrics = {config: compute_metrics(group) for config, group in grouped}

        if plot_box:
            data_to_plot = [group['roundCount'].dropna() for _, group in grouped]
            labels = list(grouped.groups.keys())
            plt.figure(figsize=(10, 6))
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Round Count')
            plt.title('Distribution of Round Count by Game Configuration Combination')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        return metrics

    else:
        # Define human-only games: both configurations contain "vs_human"
        human_games = df[
            df['gameConfigPlayer1'].str.contains("vs_human", na=False) &
            df['gameConfigPlayer2'].str.contains("vs_human", na=False)
            ]
        # Define bot-involved games: at least one configuration contains "vs_bot"
        bot_games = df[
            df['gameConfigPlayer1'].str.contains("vs_bot", na=False) |
            df['gameConfigPlayer2'].str.contains("vs_bot", na=False)
            ]

        metrics = {
            'human': compute_metrics(human_games),
            'bot': compute_metrics(bot_games),
            'total': compute_metrics(df)
        }

        if plot_box:
            data_to_plot = [
                human_games['roundCount'].dropna(),
                bot_games['roundCount'].dropna(),
                df['roundCount'].dropna()
            ]
            plt.figure(figsize=(8, 6))
            plt.boxplot(data_to_plot, labels=['Human', 'Bot', 'Total'])
            plt.ylabel('Round Count')
            plt.title('Distribution of Round Count by Game Configuration')
            plt.show()

        return metrics

# same function but without bot but player versus player rates
def calculate_game_metrics_per_player(games_df):

    # Calculate average number of rounds per player and success rate per player
    total_rounds = games_df.groupby('player1Id')['roundCount'].sum().add(games_df.groupby('player2Id')['roundCount'].sum(), fill_value=0)
    total_games_per_player = games_df['player1Id'].value_counts().add(games_df['player2Id'].value_counts(), fill_value=0)
    average_num_round = (total_rounds / total_games_per_player).fillna(0)
    average_success_rate = (games_df[games_df['status'] == 'won']['player1Id'].value_counts().add(games_df[games_df['status'] == 'won']['player2Id'].value_counts(), fill_value=0) / total_games_per_player).fillna(0)

    # Combine all metrics into a single DataFrame
    metrics_df = pd.DataFrame({
        'Average Number of Rounds': average_num_round,
        'Average Success Rate': average_success_rate
    })

    return metrics_df