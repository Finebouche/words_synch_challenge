import openai
import csv
from datetime import datetime
import re

# Get open_ai api key from open_ai_key.txt
with open('open_ai_key.txt', 'r') as file:
    openai.api_key = file.read().replace('\n', '')

def create_round_template(round_number, past_words):
    return f"\nRound {round_number}! Past words, forbidden to use are {', '.join(past_words)}. Please give your word for the current round.\n"


def get_openai_response(interaction_history, allow_thinking=True):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to gpt-3.5-turbo or another model
            messages=interaction_history,
            max_tokens=20,
            stop=None,
            temperature=0.7
        )
        return re.sub(r'[^a-zA-Z]', "", response.choices[0].message.content)
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None


def play_game():
    past_words = []
    game_data = []
    round_number = 1
    system_message = ("We are playing a game where at each round we say a word. The goal is to produce the same word "
                      "based on previous words at which point the game ends.")

    messages = [{"role": "system", "content": system_message}]

    while round_number <= 10:
        round_message = create_round_template(round_number,
                                              past_words) if past_words else ("Round 1! New game, please give your "
                                                                              "first word.")

        # Update interaction history for Player 1
        messages.append({"role": "user", "content": round_message + " Player 1: '"})
        bot1_word = get_openai_response(messages)
        messages[-1]["content"] += f"{bot1_word}'"

        # Update interaction history for Player 2
        messages.append({"role": "user", "content": f"Player 2: '"})
        bot2_word = get_openai_response(messages)
        messages[-1]["content"] += f"{bot2_word}'"

        if bot1_word is None or bot2_word is None:
            break
        ## Check if word was already used
        if bot1_word in past_words or bot2_word in past_words:
            print(f"Player 1 repeated a word in round {round_number}! Game over.")
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
