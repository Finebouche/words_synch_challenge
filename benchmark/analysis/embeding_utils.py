import openai
import gensim.downloader as api
import pandas as pd
from tqdm import tqdm

# Get open_ai api key from open_ai_key.txt
with open('../open_ai_key.txt', 'r') as file:
    openai.api_key = file.read().replace('\n', '')

def get_openai_embedding(word):
    response = openai.embeddings.create(
        input=word,
        model="text-embedding-ada-002"  # Choose the appropriate embedding model
    )
    return response.data[0].embedding

def get_openai_embeddings(words):
    embeddings = []
    for word in words:
        embeddings.append(get_openai_embedding(word))
    return embeddings


def load_model(model_name):
    """
    Load a pre-trained model by name. Supports 'word2vec' and 'glove'.
    """
    if model_name == "word2vec":
        # Load Word2Vec using Gensim's downloader
        # This will download the model if not already downloaded
        model = api.load("word2vec-google-news-300")
    elif model_name == "glove":
        # Load GloVe model, similarly using Gensim's downloader
        model = api.load("glove-wiki-gigaword-100")
    else:
        raise ValueError("Unsupported model. Choose 'word2vec' or 'glove'")
    return model

def get_embeddings(words, model):
    embeddings = []
    for word in words:
        try:
            embeddings.append(model[word.lower()].tolist())
        except KeyError:
            # Word not in vocabulary
            # Handling out-of-vocabulary words
            embeddings.append([0] * model.vector_size)  # Return zero vector if word not found
    return embeddings

def get_embeddings_for_table(games_df: pd.DataFrame, model_name="openai"):
    """
    Get embeddings for the last words played by each player in each game.
    Optionally reduce them with PCA to a smaller dimension.

    - If pca_components is not None, we fit a PCA on all the round-level embeddings
      across all games, then create new columns with the PCA-reduced embeddings.
    """
    # Check if the original embeddings are already present
    embed_col1 = f"embedding1_{model_name}"
    embed_col2 = f"embedding2_{model_name}"
    if embed_col1 in games_df.columns and embed_col2 in games_df.columns:
        print(f"Embeddings for '{model_name}' already exist in DataFrame.")
    else:
        # Otherwise, load the model if needed
        if model_name == "openai":
            embedding_model = None
        elif model_name in ("word2vec", "glove"):
            embedding_model = load_model(model_name=model_name)
        else:
            raise ValueError("Unsupported model. Choose 'openai', 'word2vec' or 'glove'")

        # We'll store new rows in a list to merge later
        embeddings_list = []
        for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Fetching Embeddings"):
            words_player1 = row['wordsPlayed1']
            words_player2 = row['wordsPlayed2']

            # Ensure both players have played words
            if len(words_player1) > 0 and len(words_player2) > 0:
                # Fetch embeddings for the last words played by each player
                if model_name == "openai":
                    embeddings_player1 = get_openai_embeddings(words_player1)
                    embeddings_player2 = get_openai_embeddings(words_player2)
                else:
                    embeddings_player1 = get_embeddings(words_player1, embedding_model)
                    embeddings_player2 = get_embeddings(words_player2, embedding_model)

                embeddings_list.append({
                    'gameId': row['gameId'],
                    embed_col1: embeddings_player1,
                    embed_col2: embeddings_player2,
                })
            else:
                # If there's no data, you can decide how to handle;
                # here we'll just store empty lists
                embeddings_list.append({
                    'gameId': row['gameId'],
                    embed_col1: [],
                    embed_col2: [],
                })

        # Merge the new embeddings back into games_df
        embeddings_df = pd.DataFrame(embeddings_list)
        games_df = games_df.merge(embeddings_df, on='gameId', how='left')
    return games_df
