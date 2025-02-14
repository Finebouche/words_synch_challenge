
if __name__ == "__main__":
    import os
    from utils.embeding_utils import (get_embeddings_for_table, calculate_pca_for_embeddings)
    from utils.game_statistics import calculate_game_metrics_per_player, calculate_game_metrics_per_configuration
    from utils.data_loading import load_sql_data, load_csv
    from strategy_analysis import strategy_analysis, plot_strategy_heatmap, print_game_turns, print_scores
    import pandas as pd

    db_name = "clean_db.db"
    csv_name = "games.csv"

    # 1) Load the data
    partial_data = ["1Ax1SPCe", "PvUHGrDy", "Ska8rixW"]
    player_ids = ["DvplG2Kz", "IfM2wWHp", "1zx9ju2S", "BVSUju5U", "lW70ICul", "OZ52imkd", "GorQHcOB", "8PSrn9JD", "u5LgXigL",
                  "1Ax1SPCe", "l8ND7njk", "sfiXibsa", "PvUHGrDy", "MO2pWpjE", "gsbTiZwO", "gDnhDODC", "Ska8rixW", "6DbWtL5r",
                  "TCFqdHBb", "wQhT1jrv"]
    dont_have_prolific_id = ["TCFqdHBb", "wQhT1jrv"]
    too_bad_ones = ["fw6dyfHt"] # 180 rounds 3 games (god paid but do not take)
    weird_one = ["QgQDS6Pw"] # 180 rounds 11 games
    if not os.path.exists(csv_name):
        players_df, games_df = load_sql_data(db_name, player_ids=player_ids)
        games_df.to_csv(csv_name, index=False)
    else:
        games_df = pd.read_csv(csv_name)

    # 2) Get embeddings (and do PCA with e.g. 50 components)
    embedding_model = "glove"
    games_df = get_embeddings_for_table( games_df, model_name=embedding_model,)
    games_df = calculate_pca_for_embeddings(
        games_df,
        model_name=embedding_model,
        num_pca_components=15,
    )

    # Save to CSV for future use
    games_df.to_csv(csv_name, index=False)

    # 3) Calculate player metrics
    # player_metrics = calculate_game_metrics_per_configuration(games_df, plot_box=True, separate_per_config=True)
    # print("Success Rate and Average Rounds for Winning Games:")
    # print(player_metrics)
    #
    # # 4) Calculate player metrics
    # player_metrics = calculate_game_metrics_per_configuration(games_df, plot_box=True, separate_per_config=False)
    # print("Average Number of Rounds and Success Rate per Player:")
    # print(player_metrics)


    # 5) Strategy analysis
    strategy_results_file = "results.csv"
    if not os.path.exists(strategy_results_file):
        results_df = strategy_analysis(games_df[:10], embedding_model, use_pca=False, use_conceptual_linking_score=False)
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
            "gameConfigPlayer1",
            "gameConfigPlayer2",
            "word_my",
            "word_opponent",
            "surveyAnswers1",
            "surveyAnswers2",
            "semantic_strategy_name",
            "quantitative_strategy_name",
            "conceptual_linking_score_my",
            "conceptual_linking_score_opponent",
            "collocation_score_my",
            "collocation_score_opponent",
        ]
        results_df_partial = results_df[cols_to_save]
        results_df_partial.to_csv(strategy_results_file, index=False)
    else:
        results_df = load_csv(
            strategy_results_file,
            columns_to_convert=["wordsPlayed1", "wordsPlayed2", "semantic_strategy_name", "quantitative_strategy_name", "conceptual_linking_score_my", "conceptual_linking_score_opponent", "collocation_score_my", "collocation_score_opponent"]
        )

    # remove outliers
    results_df = results_df[results_df['roundCount'] < 20]
    results_df = results_df[results_df['roundCount'] > 2]

    # 6) Group by configuration and compute the average scores
    # print_scores(results_df)
    print(results_df)

    plot_strategy_heatmap(results_df, strategy_col="semantic_strategy_name", groupby='gameConfig')
    plot_strategy_heatmap(results_df, strategy_col="quantitative_strategy_name", groupby='gameConfig')
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

    print_game_turns(results_df , n=15)



