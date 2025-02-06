from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import numpy as np
from embeding_utils import get_embeddings

# Function to ensure the embeddings are properly formatted as numpy arrays
def ensure_numpy_array(embeddings):
    # Convert to numpy array if not already
    return np.array(embeddings, dtype=float)


def safe_calculate_euclidean_distances(row):
    distances = calculate_euclidean_distances(row)
    # Ensure the result is a list or tuple
    if not isinstance(distances, (list, tuple)):
        distances = [distances]
    # If fewer than 2 elements, pad with NaN; if more than 2, trim the list.
    if len(distances) < 2:
        distances = list(distances) + [np.nan] * (2 - len(distances))
    elif len(distances) > 2:
        distances = list(distances)[:2]
    return pd.Series(distances)


# Function to calculate distances
def calculate_euclidean_distances(row):
    embeddings_current = get_embeddings(row['wordsPlayed1'])
    embeddings_other = get_embeddings(row['wordsPlayed2'])

    embeddings_current = ensure_numpy_array(embeddings_current)
    embeddings_other = ensure_numpy_array(embeddings_other)

    # Calculate the average embeddings of the two last words
    average_embeddings = (embeddings_current[:-1] + embeddings_other[:-1]) / 2

    # Calculate the distances to the previous word (Mirroring strategy)
    distances_to_prev = np.linalg.norm(embeddings_current[1:] - embeddings_other[:-1], axis=1)

    # Calculate the distances to the average of the two last words (Balancing strategy)
    distances_to_avg = np.linalg.norm(embeddings_current[1:] - average_embeddings, axis=1)

    return distances_to_prev, distances_to_avg

def plot_distances(embeddings_1, embeddings_2, average_embeddings):
    # Get the previous embeddings array (all but the last element)
    previous_embeddings_1 = embeddings_1[:-1] if len(embeddings_1) > 1 else []  # All but the last, empty if only one word
    previous_embeddings_2 = embeddings_2[:-1] if len(embeddings_2) > 1 else []  # All but the last, empty if only one word

    # Calculating distances
    distances_to_prev = [euclidean(embeddings_2[1:][i], previous_embeddings_1[i]) for i in range(len(previous_embeddings_1))]
    distances_to_avg = [euclidean(embeddings_2[i], average_embeddings[i]) for i in range(len(embeddings_2))]
    distance_of_words = [euclidean(embeddings_2[i], embeddings_1[i]) for i in range(len(embeddings_1))]

    # Adjusting plots
    plt.figure(figsize=(10, 5))
    plt.plot(distances_to_prev, label='Distance to Model 1 previous word', marker='o')
    plt.plot(distances_to_avg, label='Distance to the Previous average of the two last words', marker='x')
    plt.plot(distance_of_words, label='Actual distance to Model 1', marker='^')
    plt.xlabel('Word Index')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.grid(True)
    plt.show()