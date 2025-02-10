import pandas as pd


def calculate_game_metrics_per_configuration(games_df):
    """
    Returns a dictionary with the success rate and average rounds for
    humans, bots, and all (total) games.
    """

    # 1) Separate out human-only games (no bot) and bot-involved games
    human_games = games_df[games_df['botId'].isna()]   # Games without any bot
    bot_games   = games_df[games_df['botId'].notna()]  # Games with a bot

    # 2) Calculate success rate for each category
    #    success rate = (number of "won" games) / (total number of games)
    human_success_rate = (human_games['status'] == 'won').mean()
    bot_success_rate   = (bot_games['status'] == 'won').mean()
    total_success_rate = (games_df['status'] == 'won').mean()

    # 3) Calculate average rounds for each category
    human_avg_rounds = human_games['roundCount'].mean()
    bot_avg_rounds   = bot_games['roundCount'].mean()
    total_avg_rounds = games_df['roundCount'].mean()

    # 4) Combine into a dictionary
    metrics = {
        'human_success': human_success_rate,
        'bot_success': bot_success_rate,
        'total_success': total_success_rate,
        'human_average_rounds': human_avg_rounds,
        'bot_average_rounds': bot_avg_rounds,
        'total_average_rounds': total_avg_rounds
    }

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