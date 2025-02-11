import pandas as pd
import numpy as np

from benchmark.analysis.qualitative_analysis.semantic_analysis import is_hypernym, is_hyponym, is_antonym, is_synonym, is_morphological_variation, \
                                                            is_thematic_alignment, is_meronym, is_holonym, is_troponym, is_entailment
from benchmark.analysis.qualitative_analysis.conceptual_linking_analysis import conceptual_linking_score
from benchmark.analysis.qualitative_analysis.collocation_analysis import collocation_score

##############################
#  QUALITATIVE (BOOLEAN)    #
##############################

def qualitative_analysis(player_games):
    """
    For each round of a game (each row in player_games), compute the following combined measures:
      - abstraction_measure: is_hypernym(current, prev_opponent) + is_hyponym(current, prev_my)
      - contrast_measure: is_antonym(current, prev_opponent) + is_antonym(current, prev_my)
      - synonym_measure: is_synonym(current, prev_opponent) + is_synonym(current, prev_my)
      - morphological_variation_measure: is_morphological_variation(current, prev_opponent) + is_morphological_variation(current, prev_my)
      - thematic_alignment_measure: is_thematic_alignment(current, prev_opponent) + is_thematic_alignment(current, prev_my)
      - meronym_holonym_measure: is_meronym(current, prev_opponent) + is_holonym(current, prev_my)
      - troponym_entailment_measure: is_troponym(current, prev_opponent) + is_entailment(current, prev_my)
      - conceptual_linking_measure: conceptual_linking_score(current, prev_opponent) + conceptual_linking_score(current, prev_my)

    For round 0 (no previous words), np.nan is stored.
    """
    # Create columns for the final combined measures
    player_games['abstraction_measure'] = None
    player_games['contrast_measure'] = None
    player_games['synonym_measure'] = None
    player_games['morphological_variation_measure'] = None
    player_games['thematic_alignment_measure'] = None
    player_games['meronym_holonym_measure'] = None
    player_games['troponym_entailment_measure'] = None
    player_games['conceptual_linking_score'] = None
    player_games['collocation_score'] = None

    for index, game in player_games.iterrows():
        # Assume the game words are stored as strings that can be evaluated into lists.
        word_my = eval(game['word_my'])
        word_opponent = eval(game['word_opponent'])
        num_rounds = min(len(word_my), len(word_opponent))

        abstraction_list = []
        contrast_list = []
        synonym_list = []
        morph_variation_list = []
        thematic_alignment_list = []
        meronym_holonym_list = []
        troponym_entailment_list = []
        conceptual_linking_list = []
        collocation_list = []

        for i in range(num_rounds):
            if i == 0:
                # Round 0: No previous round to compare to.
                abstraction_list.append(np.nan)
                contrast_list.append(np.nan)
                synonym_list.append(np.nan)
                morph_variation_list.append(np.nan)
                thematic_alignment_list.append(np.nan)
                meronym_holonym_list.append(np.nan)
                troponym_entailment_list.append(np.nan)
                conceptual_linking_list.append(np.nan)
                collocation_list.append(np.nan)
            else:
                current_word = word_my[i]
                prev_opponent_word = word_opponent[i - 1]
                prev_my_word = word_my[i - 1]

                abstraction_score = int(is_hypernym(current_word, prev_opponent_word)) + int(
                    is_hyponym(current_word, prev_my_word))
                contrast_score = int(is_antonym(current_word, prev_opponent_word)) + int(
                    is_antonym(current_word, prev_my_word))
                synonym_score = int(is_synonym(current_word, prev_opponent_word)) + int(
                    is_synonym(current_word, prev_my_word))
                morph_variation_score = int(is_morphological_variation(current_word, prev_opponent_word)) + int(
                    is_morphological_variation(current_word, prev_my_word))
                thematic_alignment_score = int(is_thematic_alignment(current_word, prev_opponent_word)) + int(
                    is_thematic_alignment(current_word, prev_my_word))
                meronym_holonym_score = int(is_meronym(current_word, prev_opponent_word)) + int(
                    is_holonym(current_word, prev_my_word))
                troponym_entailment_score = int(is_troponym(current_word, prev_opponent_word)) + int(
                    is_entailment(current_word, prev_my_word))
                cl_score = conceptual_linking_score(current_word, prev_opponent_word, verbose=True) + \
                             conceptual_linking_score(current_word, prev_my_word, verbose=True)
                collo_score = collocation_score(current_word, prev_opponent_word) + collocation_score(current_word, prev_my_word)

                abstraction_list.append(abstraction_score)
                contrast_list.append(contrast_score)
                synonym_list.append(synonym_score)
                morph_variation_list.append(morph_variation_score)
                thematic_alignment_list.append(thematic_alignment_score)
                meronym_holonym_list.append(meronym_holonym_score)
                troponym_entailment_list.append(troponym_entailment_score)
                conceptual_linking_list.append(cl_score)
                collocation_list.append(collo_score)


        player_games.at[index, 'abstraction_measure'] = abstraction_list
        player_games.at[index, 'contrast_measure'] = contrast_list
        player_games.at[index, 'synonym_measure'] = synonym_list
        player_games.at[index, 'morphological_variation_measure'] = morph_variation_list
        player_games.at[index, 'thematic_alignment_measure'] = thematic_alignment_list
        player_games.at[index, 'meronym_holonym_measure'] = meronym_holonym_list
        player_games.at[index, 'troponym_entailment_measure'] = troponym_entailment_list
        player_games.at[index, 'conceptual_linking_score'] = conceptual_linking_list
        player_games.at[index, 'collocation_score'] = collocation_list

    return player_games


##############################
# STRATEGY ASSIGNMENT
##############################

def assign_semantic_strategy(row):
    """
    For each round, assign one or more labels based on the combined measures.
    The measures considered are:
      - abstraction_measure
      - contrast_measure
      - synonym_measure
      - morphological_variation_measure
      - thematic_alignment_measure
      - meronym_holonym_measure
      - troponym_entailment_measure
      - conceptual_linking_score
      - collocation_score

    For each round:
      - If all measures are zero, label the round with ["none"].
      - Otherwise, assign the name(s) of the measure(s) that achieved the maximum value (if > 0).
    """
    abstraction = row['abstraction_measure']
    contrast = row['contrast_measure']
    synonym = row['synonym_measure']
    morph = row['morphological_variation_measure']
    thematic = row['thematic_alignment_measure']
    part_whole = row['meronym_holonym_measure']
    manner_imply = row['troponym_entailment_measure']

    num_rounds = len(abstraction)
    strategy_labels = []

    for i in range(num_rounds):
        if pd.isna(abstraction[i]):
            strategy_labels.append(["none"])
            continue

        measure_values = {
            'synonym': synonym[i],
            'morphological_variation': morph[i],
            'abstraction': abstraction[i],
            'contrast': contrast[i],
            'thematic_alignment': thematic[i],
            'part_whole': part_whole[i],
            'manner_imply': manner_imply[i],
        }

        if all(val == 0 for val in measure_values.values()):
            strats_used = ["other"]
        else:
            max_value = max(measure_values.values())
            strats_used = [mname for mname, mval in measure_values.items() if mval == max_value and mval > 0]
            if not strats_used:
                strats_used = ["other"]

        strategy_labels.append(strats_used)


    return strategy_labels