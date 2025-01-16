import openai


# Get open_ai api key from open_ai_key.txt
with open('../open_ai_key.txt', 'r') as file:
    openai.api_key = file.read().replace('\n', '')


def get_embeddings(words):
    embeddings = []
    for word in words:
        response = openai.embeddings.create(
            input=word,
            model="text-embedding-ada-002"  # Choose the appropriate embedding model
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

