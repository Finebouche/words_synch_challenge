import requests  # For querying ConceptNet

def conceptual_linking_score(word_a, word_b, verbose=False):
    """
    Query ConceptNet for an association between word_a and word_b.
    Returns a numerical weight indicating the strength of the association.
    If no association is found, returns 0.0.

    If verbose is True, prints the URL, response, and computed score.
    """
    word_a = word_a.lower()
    word_b = word_b.lower()
    # Build the query URL. (You might need to URL-encode words if they contain spaces.)
    url = f"http://api.conceptnet.io/query?node=/c/en/{word_a}&other=/c/en/{word_b}&limit=1"

    if verbose:
        print("Querying ConceptNet:", url)

    try:
        response = requests.get(url).json()
        if verbose:
            print("Response:", response)
        edges = response.get('edges', [])
        if edges:
            # Get the maximum weight among the returned edges.
            weights = [edge.get('weight', 0) for edge in edges]
            result = max(weights)
        else:
            result = 0.0
        if verbose:
            print(f"Association score for '{word_a}' and '{word_b}':", result)
        return result
    except Exception as e:
        if verbose:
            print("Error querying ConceptNet:", e)
        return 0.0

if __name__ == "__main__":
    ##########################################
    # CONCEPTUAL LINKING examples (ConceptNet)
    ##########################################
    print("CONCEPTUAL LINKING EXAMPLES")
    # Requires an internet connection to ConceptNet
    print("Conceptual linking score between 'computer' and 'keyboard':",
          conceptual_linking_score('computer', 'keyboard', verbose=False))  # > 0 if accessible

    # More:
    print("Conceptual linking score between 'phone' and 'call':",
          conceptual_linking_score('phone', 'call', verbose=False))
    print("Conceptual linking score between 'house' and 'door':",
          conceptual_linking_score('house', 'door', verbose=False))