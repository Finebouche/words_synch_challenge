import openai
import csv
from datetime import datetime

# Set your OpenAI API key here
openai.api_key = 'your-api-key'

def create_round_template(round_number, past_words):
    return f"\nRound {round_number}! Past words, forbidden to use are {', '.join(past_words)}. Please give your word for the current round.\n"

def get_openai_response(interaction_history):
    try:
        response = openai.Completion.create(
            engine="davinci",  # You can change this to gpt-3.5-turbo or another model
            prompt=interaction_history,
            max_tokens=20,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def play_game():
    past_words = []
    rule_token = "We are playing a game where at each round we say a word. The goal is to produce the same word based on previous words at which point the game ends."
    round_number = 1
    game_data = []
    interaction_history = rule_token

    while round_number <= 10:
        if past_words:
            interaction_history += create_round_template(round_number, past_words)
        else:
            interaction_history += "\nRound 1! New game, please give your first word.\n"

        interaction_history += f"Player 1: '"
        bot1_word = get_openai_response(interaction_history)
        interaction_history += f"{bot1_word}'\nPlayer 2: '"
        bot2_word = get_openai_response(interaction_history)
        interaction_history += f"{bot2_word}'\n"

        if bot1_word is None or bot2_word is None:
            break

        past_words.extend([bot1_word, bot2_word])
        game_data.append([datetime.now(), round_number, bot1_word, bot2_word])

        if bot1_word == bot2_word:
            print(f"Game won in round {round_number}! Both bots said: {bot1_word}")
            break

        round_number += 1

    return game_data

def save_results_to_csv(results):
    with open('game_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Round', 'Bot 1 Word', 'Bot 2 Word'])
        writer.writerows(results)

if __name__ == "__main__":
    results = play_game()
    save_results_to_csv(results)