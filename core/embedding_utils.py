import tiktoken
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

client = OpenAI()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def split_text(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def generate_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding