import os
import pickle
import dotenv
import numpy as np
import faiss
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

client = OpenAI()

# Carrega dados e índice FAISS
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Olá! Sou o chatbot da Unisinos. Me pergunte algo!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_text("Pensando... 🤖")

    try:
        context_chunks = search_similar_chunks(user_text)
        context_text = "\n".join(context_chunks)

        messages = [
            {"role": "system", "content": "Você é um assistente da Universidade Unisinos, responda com base nas informações fornecidas."},
            {"role": "user", "content": f"Contexto:\n{context_text}\n\nPergunta: {user_text}"}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Erro ao processar sua pergunta: {str(e)}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot rodando...")
    app.run_polling()

if __name__ == "__main__":
    main()
