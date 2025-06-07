import os
import pickle
import dotenv
import numpy as np
from openai import OpenAI
import faiss

dotenv.load_dotenv()
client = OpenAI()

# Carrega os chunks e índice FAISS
with open("docs.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("faiss_index.index")

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def search_similar_chunks(query, k=3):
    embedding = generate_embedding(query)
    D, I = index.search(np.array([embedding]).astype("float32"), k)
    return [documents[i] for i in I[0]]

def ask_chatgpt(question):
    context_chunks = search_similar_chunks(question)
    context = "\n".join(context_chunks)

    messages = [
        {"role": "system", "content": "Você é um assistente da Universidade Unisinos, responda baseado nas informações fornecidas."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.strip()

def main():
    print("Chatbot Unisinos — digite 'sair' para encerrar")
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ("sair", "exit", "quit"):
            break

        print("Bot: pensando...")
        try:
            answer = ask_chatgpt(user_input)
            print("Bot:", answer)
        except Exception as e:
            print("Erro:", str(e))

if __name__ == "__main__":
    main()
