import pandas as pd
import nltk
import re

from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.probability import FreqDist


# Make sure you have the necessary NLTK resources:
# nltk.download('punkt')  # If you need tokenization
# nltk.download('stopwords')  # If you need to remove stopwords

def preprocess_tokens(word_list):
    """
    Optional: clean or filter words (remove punctuation, convert to lowercase, etc.).
    Modify as needed for your use case.
    """
    cleaned = []
    for w in word_list:
        w = w.lower()
        # Keep only alphabetic tokens (optional):
        if re.match(r'^[a-z]+$', w):
            cleaned.append(w)
    return cleaned


def collocation_analysis(all_words, top_n=10, freq_filter=2):
    """
    Finds top bigram collocations from a list of words using NLTK's BigramCollocationFinder.
    The measure used here is Pointwise Mutual Information (PMI).

    :param all_words: list of words (tokens) across the entire corpus.
    :param top_n: how many collocations to return.
    :param freq_filter: minimum frequency to consider a bigram.
    :return: list of top-n bigrams (tuples) by PMI.
    """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_words)
    # Exclude bigrams that occur fewer than freq_filter times
    finder.apply_freq_filter(freq_filter)
    # Return top collocations by PMI
    top_collocations = finder.nbest(bigram_measures.pmi, top_n)
    return top_collocations



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
    for idx, row in player_games.iterrows():
        my_words = eval(row['word_my'])  # e.g., "['cat', 'dog']"
        opponent_words = eval(row['word_opponent'])
        # Extend our corpus
        all_words.extend(my_words)
        all_words.extend(opponent_words)

    # Optional: preprocess (remove punctuation, lower, etc.)
    all_words = preprocess_tokens(all_words)

    # Get top collocations
    top_collocs = collocation_analysis(all_words, top_n=10, freq_filter=2)

    return {
        'top_collocations': top_collocs,
    }


if __name__ == "__main__":
    # Example usage: We'll create a small dummy dataset.
    data = {
        # Each row's string is a list of tokens (like in your original code).
        'word_my': ["['cat', 'dog']", "['tree', 'oak', 'forest']", "['phone', 'call', 'text']"],
        'word_opponent': ["['dog', 'cat']", "['oak', 'wood', 'plant']", "['phone', 'call', 'talk']"]
    }
    df = pd.DataFrame(data)

    # Run the collocation & distribution analysis
    results = collocation_distribution_analysis(df)

    # Print the findings
    print("Top Collocations by PMI:", results['top_collocations'])
