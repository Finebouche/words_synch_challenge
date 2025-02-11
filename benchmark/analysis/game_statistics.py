import pandas as pd
import matplotlib.pyplot as plt

def calculate_game_metrics_per_configuration(games_df, plot_box=False):
    """
    Returns a structured dictionary with the success rate, average rounds,
    and their standard deviations for human-only games, bot-involved games,
    and all (total) games. Optionally plots the round count distributions as box plots.

    Parameters:
        games_df (DataFrame): The dataframe containing game data with columns
                              'botId', 'status', and 'roundCount'.
        plot_box (bool): If True, a box plot of round counts by configuration is displayed.

    Returns:
        dict: A dictionary with metrics structured by configuration.
    """
    # 1) Separate out human-only games (no bot) and bot-involved games
    human_games = games_df[games_df['botId'].isna()]  # Games without any bot
    bot_games = games_df[games_df['botId'].notna()]  # Games with a bot

    # 2) Calculate success rates and standard deviations
    # Convert the boolean series to floats to compute std
    human_success_series = (human_games['status'] == 'won').astype(float)
    bot_success_series = (bot_games['status'] == 'won').astype(float)
    total_success_series = (games_df['status'] == 'won').astype(float)

    human_success_rate = human_success_series.mean()
    bot_success_rate = bot_success_series.mean()
    total_success_rate = total_success_series.mean()

    human_success_std = human_success_series.std()
    bot_success_std = bot_success_series.std()
    total_success_std = total_success_series.std()

    # 3) Calculate average rounds and standard deviations for each category
    human_avg_rounds = human_games['roundCount'].mean()
    bot_avg_rounds = bot_games['roundCount'].mean()
    total_avg_rounds = games_df['roundCount'].mean()

    human_rounds_std = human_games['roundCount'].std()
    bot_rounds_std = bot_games['roundCount'].std()
    total_rounds_std = games_df['roundCount'].std()

    # 4) Combine all metrics into a structured dictionary grouped by configuration
    metrics = {
        'human': {
            'success_rate': human_success_rate,
            'success_std': human_success_std,
            'average_rounds': human_avg_rounds,
            'rounds_std': human_rounds_std
        },
        'bot': {
            'success_rate': bot_success_rate,
            'success_std': bot_success_std,
            'average_rounds': bot_avg_rounds,
            'rounds_std': bot_rounds_std
        },
        'total': {
            'success_rate': total_success_rate,
            'success_std': total_success_std,
            'average_rounds': total_avg_rounds,
            'rounds_std': total_rounds_std
        }
    }

    # 5) Optionally, plot the round count distributions as box plots
    if plot_box:
        data_to_plot = [
            human_games['roundCount'].dropna(),
            bot_games['roundCount'].dropna(),
            games_df['roundCount'].dropna()
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