import pandas as pd


def calculate_game_metrics_per_player(games_df):
    # Filter games where status is 'won' to calculate success rates
    won_games = games_df[games_df['status'] == 'won']

    # Calculate human_success_rate
    human_games = won_games[won_games['botId'].isna()]  # Games without a bot involved
    human_success_count = human_games['player1Id'].value_counts().add(human_games['player2Id'].value_counts(), fill_value=0)
    total_human_games = games_df[games_df['botId'].isna()]['player1Id'].value_counts().add(games_df[games_df['botId'].isna()]['player2Id'].value_counts(), fill_value=0)
    human_success_rate = (human_success_count / total_human_games).fillna(0)

    # Calculate bot_success_rate
    bot_games = won_games[won_games['player2Id'].isna()]  # Assuming bots only play in player2Id's slot
    bot_success_count = bot_games['player1Id'].value_counts()
    total_bot_games = games_df[games_df['player2Id'].isna()]['player1Id'].value_counts()
    bot_success_rate = (bot_success_count / total_bot_games).fillna(0)

    # Calculate average number of rounds per player
    total_rounds = games_df.groupby('player1Id')['roundCount'].sum().add(games_df.groupby('player2Id')['roundCount'].sum(), fill_value=0)
    total_games_per_player = games_df['player1Id'].value_counts().add(games_df['player2Id'].value_counts(), fill_value=0)
    average_num_round = (total_rounds / total_games_per_player).fillna(0)

    # Combine all metrics into a single DataFrame
    metrics_df = pd.DataFrame({
        'Human Success Rate': human_success_rate,
        'Bot Success Rate': bot_success_rate,
        'Average Number of Rounds': average_num_round
    })

    return metrics_df
