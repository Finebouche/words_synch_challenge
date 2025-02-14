import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()


def is_hypernym(candidate_word, original_word):
    """
    Return True if any synset of `candidate_word` appears in
    the hypernym paths of any synset of `original_word`.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    for o_syn in original_synsets:
        for path in o_syn.hypernym_paths():
            if any(c_syn in path for c_syn in candidate_synsets):
                return True
    return False

def is_hyponym(candidate_word, original_word):
    """
    Return True if candidate_word is a hyponym of original_word.
    i.e., original_word is a hypernym of candidate_word.
    """
    return is_hypernym(original_word, candidate_word)

def is_antonym(word_a, word_b):
    """
    Return True if word_a and word_b are antonyms in WordNet.
    Checks both directions: word_a -> word_b OR word_b -> word_a.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()

    # 1) Check if word_b is an antonym of word_a
    for syn in wn.synsets(word_a):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                if ant.name().lower() == word_b:
                    return True

    # 2) Check if word_a is an antonym of word_b
    for syn in wn.synsets(word_b):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                if ant.name().lower() == word_a:
                    return True

    return False


def get_wordnet_pos(treebank_tag):
    """
    Convert POS tag from the Penn Treebank or similar to a
    simplified WordNet tag: (n)oun, (v)erb, (a)djective, (r)adverb.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def is_morphological_variation(word_a, word_b):
    # Step 1: POS-tag the words
    tokens = [word_a, word_b]
    tagged = nltk.pos_tag(tokens)  # e.g. [('stronger', 'JJR'), ('strong', 'JJ')]

    w_a, pos_a = tagged[0]
    w_b, pos_b = tagged[1]

    # Step 2: Convert the Treebank tag to a WordNet tag
    wn_pos_a = get_wordnet_pos(pos_a) or wordnet.NOUN  # fallback = NOUN
    wn_pos_b = get_wordnet_pos(pos_b) or wordnet.NOUN

    # Step 3: Lemmatize with the correct POS
    lem_a = lemmatizer.lemmatize(w_a.lower(), wn_pos_a)
    lem_b = lemmatizer.lemmatize(w_b.lower(), wn_pos_b)

    # Step 4: Check if they share the same lemma, but differ as strings
    return (lem_a == lem_b)

def is_synonym(word_a, word_b):
    """
    Return True if word_a and word_b share at least one synset in WordNet.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    synsets_a = set(wn.synsets(word_a))
    synsets_b = set(wn.synsets(word_b))
    # If there's any overlap, it indicates a shared sense (synonym in at least one sense).
    return len(synsets_a.intersection(synsets_b)) > 0



def is_meronym(candidate_word, original_word):
    """
    Return True if candidate_word is a meronym of original_word.
    Checks if any synset of candidate_word appears in any of the meronym lists
    (part, substance, or member meronyms) of any synset of original_word.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    for o_syn in original_synsets:
        for meronym_func in [lambda s: s.part_meronyms(),
                             lambda s: s.substance_meronyms(),
                             lambda s: s.member_meronyms()]:
            meronyms = meronym_func(o_syn)
            if any(c_syn in meronyms for c_syn in candidate_synsets):
                return True
    return False


def is_holonym(candidate_word, original_word):
    """
    Return True if candidate_word is a holonym of original_word.
    Checks if any synset of candidate_word appears in any of the holonym lists
    (part, substance, or member holonyms) of any synset of original_word.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word)
    original_synsets = wn.synsets(original_word)

    for o_syn in original_synsets:
        for holonym_func in [lambda s: s.part_holonyms(),
                             lambda s: s.substance_holonyms(),
                             lambda s: s.member_holonyms()]:
            holonyms = holonym_func(o_syn)
            if any(c_syn in holonyms for c_syn in candidate_synsets):
                return True
    return False


def is_troponym(candidate_word, original_word):
    """
    Return True if candidate_word is a troponym of original_word.
    (Troponyms capture the manner of performing a verb.)
    Applicable for verbs.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word, pos=wn.VERB)
    original_synsets = wn.synsets(original_word, pos=wn.VERB)

    for o_syn in original_synsets:
        # Use try/except to handle if 'troponyms' is not available
        try:
            troponyms = o_syn.troponyms()
        except AttributeError:
            troponyms = []  # Fallback if not implemented
        if any(c_syn in troponyms for c_syn in candidate_synsets):
            return True
    return False


def is_entailment(candidate_word, original_word):
    """
    Return True if candidate_word is entailed by original_word.
    (Entailment in verbs indicates that the occurrence of one action implies another.)
    Applicable for verbs.
    """
    candidate_word = candidate_word.lower()
    original_word = original_word.lower()
    candidate_synsets = wn.synsets(candidate_word, pos=wn.VERB)
    original_synsets = wn.synsets(original_word, pos=wn.VERB)

    for o_syn in original_synsets:
        entailments = o_syn.entailments()
        if any(c_syn in entailments for c_syn in candidate_synsets):
            return True
    return False


def share_broad_category(word_a, word_b, depth_threshold=2):
    """
    Returns True if `word_a` and `word_b` share a common hypernym
    within `depth_threshold` levels up the hypernym tree.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    syns_a = wn.synsets(word_a)
    syns_b = wn.synsets(word_b)

    for syn_a in syns_a:
        paths_a = syn_a.hypernym_paths()
        up_a = set()
        for path in paths_a:
            for i in range(len(path)-1, max(-1, len(path)-1 - depth_threshold), -1):
                up_a.add(path[i])

        for syn_b in syns_b:
            paths_b = syn_b.hypernym_paths()
            up_b = set()
            for path in paths_b:
                for i in range(len(path)-1, max(-1, len(path)-1 - depth_threshold), -1):
                    up_b.add(path[i])

            if up_a.intersection(up_b):
                return True
    return False

def is_thematic_alignment(word_a, word_b):
    return share_broad_category(word_a, word_b, depth_threshold=2)


if __name__ == "__main__":
    ##########################################
    # HYPERNYM / HYPONYM examples
    ##########################################
    print("HYPERNYM / HYPONYM EXAMPLES")
    # Expected: True if WordNet has these mapped
    print("Is 'animal' a hypernym of 'dog'? :", is_hypernym('animal', 'dog'))  # True
    print("Is 'dog' a hyponym of 'animal'? :", is_hyponym('dog', 'animal'))    # True

    # Additional:
    print("Is 'vehicle' a hypernym of 'car'? :", is_hypernym('vehicle', 'car'))  # True
    print("Is 'car' a hyponym of 'vehicle'? :", is_hyponym('car', 'vehicle'))    # True

    # You might try something more specialized:
    print("Is 'tree' a hypernym of 'oak'? :", is_hypernym('tree', 'oak'))        # Often True
    print("Is 'oak' a hyponym of 'tree'? :", is_hyponym('oak', 'tree'))          # Often True

    ##########################################
    # ANTONYM examples
    ##########################################
    print("ANTONYM EXAMPLES")
    # Basic known antonyms:
    print("Are 'hot' and 'cold' antonyms? :", is_antonym('hot', 'cold'))         # True
    print("Are 'big' and 'small' antonyms? :", is_antonym('big', 'small'))       # True
    print("Are 'happy' and 'unhappy' antonyms? :", is_antonym('happy', 'unhappy'))  # True
    print("Are 'happy' and 'sad' antonyms? :", is_antonym('happy', 'sad'))  # False

    ##########################################
    # SYNONYM examples
    ##########################################
    print("SYNONYM EXAMPLES")
    # Basic synonyms:
    print("Are 'big' and 'large' synonyms? :", is_synonym('big', 'large'))       # True
    print("Are 'car' and 'automobile' synonyms? :", is_synonym('car', 'automobile'))  # True

    # More examples:
    print("Are 'quick' and 'fast' synonyms? :", is_synonym('quick', 'fast'))     # True
    print("Are 'sea' and 'ocean' synonyms? :", is_synonym('sea', 'ocean'))       # Sometimes True in WordNet

    ##########################################
    # MORPHOLOGICAL VARIATION examples
    ##########################################
    print("MORPHOLOGICAL VARIATION EXAMPLES")
    # Different inflections or forms of the same lemma:
    print("Are 'stronger' and 'strong' morphological variations? :",
          is_morphological_variation('stronger', 'strong'))  # True

    # More examples:
    print("Are 'running' and 'run' morphological variations? :",
          is_morphological_variation('running', 'run'))      # True
    print("Are 'eating' and 'eat' morphological variations? :",
          is_morphological_variation('eating', 'eat'))       # True

    ##########################################
    # THEMATIC ALIGNMENT examples
    ##########################################
    print("THEMATIC ALIGNMENT EXAMPLES")
    # Check if words share broad categories (same hypernyms near the top)
    print("Do 'pen' and 'pencil' share a broad category? :",
          is_thematic_alignment('pen', 'pencil'))            # True
    print("Do 'cat' and 'dog' share a broad category? :",
          is_thematic_alignment('cat', 'dog'))               # True (both are animals)
    print("Do 'car' and 'motorcycle' share a broad category? :",
          is_thematic_alignment('car', 'motorcycle'))        # True (both vehicles)

    # Might or might not work if WordNet sees them in different branches:
    print("Do 'desk' and 'table' share a broad category? :",
          is_thematic_alignment('desk', 'table'))            # Often True (both furniture)

    ##########################################
    # MERONYM / HOLONYM examples
    ##########################################
    print("MERONYM / HOLONYM EXAMPLES")
    # "finger" is part of a "hand"
    print("Is 'finger' a meronym (part) of 'hand'? :",
          is_meronym('finger', 'hand'))                      # True
    print("Is 'hand' a holonym (whole) of 'finger'? :",
          is_holonym('hand', 'finger'))                      # True

    # Additional:
    print("Is 'wheel' a meronym (part) of 'car'? :",
          is_meronym('wheel', 'car'))                        # True (often recognized)
    print("Is 'car' a holonym (whole) of 'wheel'? :",
          is_holonym('car', 'wheel'))                        # True

    # Another:
    print("Is 'room' a meronym (part) of 'house'? :",
          is_meronym('room', 'house'))                       # Possibly True
    print("Is 'house' a holonym (whole) of 'room'? :",
          is_holonym('house', 'room'))                       # Possibly True

    ##########################################
    # TROPONYMS (verb manner) / ENTAILMENT examples
    ##########################################
    print("TROPONYM / ENTAILMENT EXAMPLES")
    # WordNet is often limited here:
    print("Is 'whisper' a troponym of 'speak'? :",
          is_troponym('whisper', 'speak'))  # Might be False in standard WordNet
    print("Is 'snore' entailed by 'sleep'? :",
          is_entailment('snore', 'sleep'))  # Also might be False

    # Additional troponym tests:
    print("Is 'amble' a troponym of 'walk'? :",
          is_troponym('amble', 'walk'))     # Sometimes True if defined
    print("Is 'limp' a troponym of 'walk'? :",
          is_troponym('limp', 'walk'))      # Possibly True

    # Additional entailment tests:
    print("Is 'buy' entailed by 'pay'? :",
          is_entailment('buy', 'pay'))      # Possibly False or True, depending on WordNet
    print("Is 'fall_asleep' entailed by 'lie_down'? :",
          is_entailment('fall_asleep', 'lie_down'))  # Possibly undefined