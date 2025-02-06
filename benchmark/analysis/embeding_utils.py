import numpy as np
import openai
import gensim.downloader as api

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

