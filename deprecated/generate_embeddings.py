import pdfplumber
from openai import OpenAI
import faiss
import pickle
import os
import dotenv
import tiktoken
import numpy as np

dotenv.load_dotenv()

client = OpenAI()

PDF_PATH = "unisinos_faq.pdf"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 100

def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))

def split_text(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def main():
    print("🔍 Lendo PDF...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    print("📐 Quebrando em pedaços...")
    chunks = split_text(raw_text)

    print(f"🧠 Gerando embeddings para {len(chunks)} chunks...")
    embeddings = [generate_embedding(chunk) for chunk in chunks]

    # Salva os textos
    with open("docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Cria índice FAISS
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, "faiss_index.index")
    print("✅ Embeddings salvos com sucesso!")

if __name__ == "__main__":
    main()
