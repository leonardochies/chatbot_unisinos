import numpy as np
from core.openai_client import generate_embedding
from data.loader import index, documents

def search_similar_chunks(query, k=3):
    embedding = generate_embedding(query)
    D, I = index.search(np.array([embedding]).astype("float32"), k)
    return [documents[i] for i in I[0]]