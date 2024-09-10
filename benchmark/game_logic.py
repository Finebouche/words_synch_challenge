import openai
import csv
from datetime import datetime
import re
import requests

# Get open_ai api key from open_ai_key.txt
with open('open_ai_key.txt', 'r') as file:
    openai.api_key = file.read().replace('\n', '')


def check_word_existence(word, language):
    endpoint = f'https://{language}.wiktionary.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': word.lower(),
        'origin': '*'
    }

    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        return '-1' not in data['query']['pages']
    except Exception as error:
        print(f'Error fetching data from Wiktionary: {error}')
        return False  # Return False in case of an error


def get_openai_response(interaction_history, seed=None, allow_thinking=True):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to gpt-3.5-turbo or another model
            messages=interaction_history,
            seed=seed,
            max_tokens=20,
            stop=None,
            temperature=1.2,
        )
        print(response.choices[0].message.content)
        return re.sub(r'[^a-zA-Z]', "", response.choices[0].message.content)
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def create_round_template(round_number, past_words, word):
    if word is None and past_words == []:
        return "Round 1. New game, please give your first (really random) word and only that word."
    else:
        return (f"{word}! We said different words, let's do another round then and try to get closer. Past words, "
                f"forbidden to use are [{', '.join(past_words)}]. Please give only your word for this round and I will "
                f"give you mine.")


def play_game():
    past_words = []
    past_words_1 = []
    past_words_2 = []
    game_data = []
    round_number = 1
    system_message = ("You are playing a game where at each round both player say a word. The goal is to produce the "
                      "same word based on previous words at which point the game ends.")

    messages1 = [{"role": "user", "content": system_message}]
    messages2 = [{"role": "user", "content": system_message}]

    status = "loses, too many rounds"
    previous_bot1_word = None
    previous_bot2_word = None
    while round_number <= 10:
        # Update interaction history for Player 1
        messages1.append({"role": "user", "content":  create_round_template(round_number, past_words, previous_bot2_word)})
        bot1_word = get_openai_response(messages1)
        messages1.append({"role": "assistant", "content": f"{bot1_word}"})

        # Update interaction history for Player 2
        messages2.append({"role": "user", "content":  create_round_template(round_number, past_words, previous_bot1_word)})
        bot2_word = get_openai_response(messages2)
        messages2.append({"role": "assistant", "content": f"{bot2_word}"})

        previous_bot1_word = bot1_word
        previous_bot2_word = bot2_word

        if bot1_word is None or bot2_word is None:
            status = "error"
            break

        if not check_word_existence(bot1_word, 'en') or not check_word_existence(bot2_word, 'en'):
            print(f"Player used a non-existing word in round {round_number}! Game over.")
            status = "loses, not existing word"
            break

        # Cancel if word was already used
        if bot1_word in past_words or bot2_word in past_words:
            print(f"Player repeated a word in round {round_number}! Game over.")
            status = "loses, repeated word"
            break

        past_words.extend([bot1_word, bot2_word])
        past_words_1.append(bot1_word)
        past_words_2.append(bot2_word)

        if bot1_word == bot2_word:
            print(f"Game won in round {round_number}! Both bots said: {bot1_word}")
            status = "wins"
            break

        round_number += 1

    game_data.append([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), round_number, status, past_words_1, past_words_2])
    return game_data


def save_results_to_csv(results):
    # Open the file in append mode; this ensures data is added to the end of the file without deleting existing content
    with open('game_results.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Optional: Only write headers if the file is newly created
        file.seek(0, 2)  # Move the cursor to the end of the file
        if file.tell() == 0:  # Check if the file is empty
            writer.writerow(['Timestamp', 'Round', 'Status', 'Past words player 1', 'Past words player 2'])  # Write headers only if the file is empty

        # Write the data rows
        writer.writerows(results)
