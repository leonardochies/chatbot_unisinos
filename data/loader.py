import pickle
import faiss

# Carrega os documentos e índice FAISS
with open("docs.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("faiss_index.index")