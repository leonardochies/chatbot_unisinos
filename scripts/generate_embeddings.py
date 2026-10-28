import pickle
import faiss
import numpy as np
from data.pdf_utils import extract_text_from_pdf
from core.embedding_utils import split_text, generate_embedding

PDF_PATH = "scripts/unisinos_faq.pdf"
DOCS_FILE = "docs.pkl"
INDEX_FILE = "faiss_index.index"

def main():
    print("🔍 Lendo PDF...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("📐 Quebrando em pedaços...")
    chunks = split_text(raw_text)

    print(f"🧠 Gerando embeddings para {len(chunks)} chunks...")
    embeddings = [generate_embedding(chunk) for chunk in chunks]

    # Salva os textos
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    # Cria índice FAISS
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_FILE)
    print("✅ Embeddings salvos com sucesso!")

if __name__ == "__main__":
    main()