import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import brown

nltk.download('brown')  # if not already downloaded
words = list(brown.words())
finder = BigramCollocationFinder.from_words(words)
bigram_measures = BigramAssocMeasures()

def collocation_score(w1, w2):
    score = finder.score_ngram(bigram_measures.likelihood_ratio, w1, w2)
    if score is None:
        return 0
    return score

def collocation_distribution_analysis(player_games):
    """
    1. Collects all words from 'word_my' and 'word_opponent' columns
    2. (Optionally) preprocesses them
    3. Runs distribution and collocation analysis across the entire dataset.

    :param player_games: Pandas DataFrame with at least these columns:
                        - 'word_my': a string that can be eval()'d into a list of words
                        - 'word_opponent': likewise
    :return: A dictionary containing:
             - 'top_collocations': the top bigrams by PMI
    """
    all_words = []

    # Gather all tokens from each row/game
    for index, game in player_games.iterrows():
        # The game words are stored as strings that represent lists, so we eval them.
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])
        # Use the smaller number of rounds, in case one list is longer than the other.
        num_rounds = min(len(word_my), len(word_opponent))

        for i in range(num_rounds):
            # Skip round 0 since there are no previous words.
            if i == 0:
                continue

            current_word = word_my[i]
            previous_my = word_my[i - 1]
            previous_opponent = word_opponent[i - 1]

            # Check if current word is collocated with either previous word.
            if collocation_score(previous_my, current_word) or collocation_score(previous_opponent, current_word):
                print(f"Game {index} - Round {i}: '{current_word}' IS collocated with a previous word")
            else:
                print(f"Game {index} - Round {i}: '{current_word}' is NOT collocated with the previous words")



if __name__ == "__main__":
    # create an example to test is_collocated
    w1 = "dog"
    w2 = "cat"
    print(collocation_score(w1, w2))  # False
    w1 = "dog"
    w2 = "bark"
    print(collocation_score(w1, w2))  # False
    w1 = "phone"
    w2 = "call"
    print(collocation_score(w1, w2))  # True