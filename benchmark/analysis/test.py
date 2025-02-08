import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def is_hypernym(candidate_word, original_word):
    """
    Return True if candidate_word is a hypernym of original_word.
    That is, if candidate_word appears in any hypernym path of original_word.
    """
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)
    for orig_syn in original_synsets:
        for path in orig_syn.hypernym_paths():
            if any(c_syn in path for c_syn in candidate_synsets):
                return True
    return False



def is_hyponym(candidate_word, original_word):
    """
    Return True if `candidate_word` is a hyponym of `original_word`.
    That is effectively the same as asking if `original_word` is a hypernym of `candidate_word`.
    """
    return is_hypernym(original_word, candidate_word)


def is_antonym(word_a, word_b):
    syns_a = wn.synsets(word_a)
    for syn in syns_a:
        for lemma in syn.lemmas():
            if lemma.antonyms():
                # collect lemma.antonyms()
                if word_b in [ant.name() for ant in lemma.antonyms()]:
                    return True
    return False

def is_synonym(word_a, word_b):
    syns_a = wn.synsets(word_a)
    synonyms_a = set()
    for syn in syns_a:
        for lemma in syn.lemmas():
            synonyms_a.add(lemma.name())
    return word_b in synonyms_a

if __name__ == "__main__":
    print(int(is_hypernym("animal", "dogs")))   # True
    print(is_hypernym("animals", "cat"))   # True
    print(is_hypernym("dog", "animal"))   # False
    print(is_hypernym("cat", "dog"))      # False


    print(is_hyponym("dog", "animal"))    # True
    print(is_hyponym("cat", "animal"))    # True
    print(is_hyponym("animal", "dog"))    # False
    print(is_hyponym("dog", "cat"))       # False