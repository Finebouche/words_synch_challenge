import openai
import gensim.downloader as api
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA

# Get open_ai api key from open_ai_key.txt
with open('../../open_ai_key.txt', 'r') as file:
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


def calculate_pca_for_embeddings(games_df: pd.DataFrame, model_name="openai", num_pca_components=None):

    embed_col1 = f"embedding1_{model_name}"
    embed_col2 = f"embedding2_{model_name}"

    # Check that the embeddings columns exist
    if embed_col1 not in games_df.columns or embed_col2 not in games_df.columns:
        raise ValueError(
            f"Embeddings not found in the DataFrame. Columns '{embed_col1}' or '{embed_col2}' missing. "
            f"Make sure you ran 'get_embeddings_for_table' first."
        )
    #    We create new columns: e.g. 'embedding1_glove_pca'
    new_col1 = f"{embed_col1}_pca"
    new_col2 = f"{embed_col2}_pca"

    # If user specifies a number of PCA components, reduce the dimension.
    if num_pca_components is not None:
        print(f"Performing PCA to reduce embeddings to {num_pca_components} dimensions.")

        # 1) Collect *all* round-level embeddings across all games for both players
        #    in a big list. We'll store them separately as:
        all_vectors = []  # shape: (N * R, D)  where R is #rounds in a game, D is original dimension

        # We'll also need to keep an index (game, 'player1'/'player2', round_id)
        # so we can reconstruct the data after PCA transform
        index_info = []

        for idx, row in games_df.iterrows():
            emb1 = eval(row.get(embed_col1, []))
            emb2 = eval(row.get(embed_col2, []))

            # Convert to np.array if not empty
            emb1_arr = np.array(emb1, dtype=float) if len(emb1) > 0 else np.empty((0, 0))
            emb2_arr = np.array(emb2, dtype=float) if len(emb2) > 0 else np.empty((0, 0))

            # Player1
            for r_i in range(emb1_arr.shape[0]):
                all_vectors.append(emb1_arr[r_i])
                index_info.append((idx, 'player1', r_i))
            # Player2
            for r_i in range(emb2_arr.shape[0]):
                all_vectors.append(emb2_arr[r_i])
                index_info.append((idx, 'player2', r_i))

        # Convert all_vectors to numpy array
        if len(all_vectors) > 0:
            all_vectors_np = np.array(all_vectors, dtype=float)
        else:
            all_vectors_np = np.empty((0, 0))

        # 2) Fit PCA
        if all_vectors_np.shape[0] > 0:
            pca = PCA(n_components=num_pca_components)
            pca.fit(all_vectors_np)

            # 3) Transform all vectors
            transformed_embeddings = pca.transform(all_vectors_np)

            # 4) Rebuild them into lists-of-rounds for each row/player
            # We'll store the new columns in memory and then assign to DataFrame at the end
            # to avoid repeated rewriting of rows.
            new_emb1_series = [[] for _ in range(len(games_df))]
            new_emb2_series = [[] for _ in range(len(games_df))]

            for (df_idx, player, round_idx), pca_vec in zip(index_info, transformed_embeddings):
                if player == 'player1':
                    new_emb1_series[df_idx].append(pca_vec.tolist())
                else:
                    new_emb2_series[df_idx].append(pca_vec.tolist())

            # Now save these lists into games_df
            games_df[new_col1] = new_emb1_series
            games_df[new_col2] = new_emb2_series

        else:
            # If no data, just create empty columns
            new_col1 = f"{embed_col1}_pca"
            new_col2 = f"{embed_col2}_pca"
            games_df[new_col1] = [[] for _ in range(len(games_df))]
            games_df[new_col2] = [[] for _ in range(len(games_df))]

    return games_df

