import random
import statistics


def run_game(vocab_A, vocab_B, max_rounds=16):
    """
    Simulates one game of the WSC-inspired experiment.

    Parameters:
      vocab_A (list of int): Vocabulary for agent A1.
      vocab_B (list of int): Vocabulary for the partner (A1 or A2).
      max_rounds (int): Maximum rounds allowed per game.

    Returns:
      win (bool): True if the agents converge (choose the same integer).
      rounds_played (int): The number of rounds played.
      choices (list of tuple): A list of (choice_A, choice_B) per round.
    """
    # Copy the vocabularies so the original lists are not altered.
    available_A = vocab_A.copy()
    available_B = vocab_B.copy()
    choices = []

    for round_num in range(max_rounds):
        # If either vocabulary is empty, the game stops.
        if not available_A or not available_B:
            break
        # Both agents pick uniformly at random from their available vocabularies.
        choice_A = random.choice(available_A)
        choice_B = random.choice(available_B)
        choices.append((choice_A, choice_B))

        # Remove the chosen words to enforce the "no repeats" rule.
        available_A.remove(choice_A)
        available_B.remove(choice_B)

        # Check for convergence.
        if choice_A == choice_B:
            return True, round_num + 1, choices

    # Game did not converge within max_rounds.
    return False, max_rounds, choices


def run_simulation(setting, n_games=100, max_rounds=16):
    """
    Runs a series of games for a given setting (S1 or S2) and computes metrics.

    Parameters:
      setting (str): Either 'S1' for A1 vs. A1 or 'S2' for A1 vs. A2.
      n_games (int): Number of games to simulate.
      max_rounds (int): Maximum rounds per game.

    Returns:
      win_rate (float): Percentage of games in which convergence occurred.
      avg_rounds (float): Average rounds used in successful games.
      avg_partner_diff (float): Average absolute difference between A1's choice and partner's previous choice.
      avg_self_diff (float): Average absolute difference between A1's choice and their own previous choice.
    """
    # Define vocabularies:
    vocab_A1 = list(range(1, 11))  # A1: integers from 1 to 10
    vocab_A2 = list(range(3, 13))  # A2: integers from 3 to 12

    wins = 0
    rounds_success = []
    partner_diffs = []  # difference between A1's current choice and partner's previous round choice
    self_diffs = []  # difference between A1's current choice and A1's previous choice

    for _ in range(n_games):
        if setting == 'S1':
            win, rounds_played, choices = run_game(vocab_A1, vocab_A1, max_rounds)
        elif setting == 'S2':
            win, rounds_played, choices = run_game(vocab_A1, vocab_A2, max_rounds)
        else:
            raise ValueError("Setting must be 'S1' or 'S2'")

        if win:
            wins += 1
            rounds_success.append(rounds_played)

        # Compute the differences from round 2 onward.
        for i in range(1, len(choices)):
            prev_partner_choice = choices[i - 1][1]  # partner's previous choice
            curr_choice_A1 = choices[i][0]  # A1's current choice
            partner_diffs.append(abs(curr_choice_A1 - prev_partner_choice))

            prev_choice_A1 = choices[i - 1][0]
            self_diffs.append(abs(curr_choice_A1 - prev_choice_A1))

    win_rate = (wins / n_games) * 100
    avg_rounds = statistics.mean(rounds_success) if rounds_success else None
    avg_partner_diff = statistics.mean(partner_diffs) if partner_diffs else None
    avg_self_diff = statistics.mean(self_diffs) if self_diffs else None

    return win_rate, avg_rounds, avg_partner_diff, avg_self_diff


# Run simulations for both settings.
for setting in ['S1', 'S2']:
    win_rate, avg_rounds, avg_partner_diff, avg_self_diff = run_simulation(setting, n_games=100, max_rounds=16)
    print(f"Results for {setting}:")
    print(f"  Win rate: {win_rate:.2f}%")
    print(f"  Average rounds (successful games): {avg_rounds:.2f}")
    print(f"  Average partner difference: {avg_partner_diff:.2f}")
    print(f"  Average self difference: {avg_self_diff:.2f}\n")