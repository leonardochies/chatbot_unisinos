"""
Bibliotecas necessárias:
- openai
- python-telegram-bot
- pdfplumber
- sqlite3 + sqlite_vector
"""

import os
import sqlite3
import pdfplumber
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import openai
import hashlib
import json

# Configurações iniciais
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
DB_PATH = 'embeddings.db'
CHUNK_SIZE = 500  # caracteres por chunk
TOP_K = 4

openai.api_key = OPENAI_API_KEY

# --- Funções de processamento de PDF e embeddings ---

def init_db(db_path=DB_PATH):
    """Inicializa o banco SQLite e cria tabela de embeddings."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id TEXT PRIMARY KEY,
        chunk_text TEXT,
        embedding BLOB
    )""")
    conn.commit()
    conn.close()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_and_chunk_pdf(pdf_path: str, chunk_size=CHUNK_SIZE):
    """Extrai texto do PDF e divide em chunks."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            i = 0
            while i < len(text):
                chunk = text[i:i+chunk_size]
                chunks.append(chunk)
                i += chunk_size
    return chunks


def embed_text(text: str):
    """Gera embedding usando text-embedding-ada-002."""
    resp = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return resp['data'][0]['embedding']


def store_chunks(chunks, db_path=DB_PATH):
    """Armazena chunks e embeddings no DB (caso não existam)."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for chunk in chunks:
        key = hash_text(chunk)
        c.execute("SELECT 1 FROM docs WHERE id=?", (key,))
        if c.fetchone():
            continue  # já existe
        emb = embed_text(chunk)
        emb_blob = json.dumps(emb)
        c.execute("INSERT INTO docs (id, chunk_text, embedding) VALUES (?, ?, ?)", (key, chunk, emb_blob))
    conn.commit()
    conn.close()

# --- Função de RAG: recuperação + geração ---

def retrieve_relevant(query: str, k=TOP_K, db_path=DB_PATH):
    """Recupera os k chunks mais similares à query."""
    q_emb = embed_text(query)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT chunk_text, embedding FROM docs")
    rows = c.fetchall()
    # Calcular similaridade coseno manualmente
    def cosine(a, b):
        import math
        dot = sum(x*y for x,y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        return dot/(na*nb)

    sims = []
    for text, emb_blob in rows:
        emb = json.loads(emb_blob)
        sims.append((cosine(q_emb, emb), text))
    sims.sort(reverse=True, key=lambda x: x[0])
    conn.close()
    return [text for _, text in sims[:k]]


def generate_answer(query: str):
    """Gera resposta com GPT-3.5-Turbo usando contexto RAG."""
    chunks = retrieve_relevant(query)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"Você é um assistente para novos alunos. Use o contexto abaixo para responder:\n{context}\n\nPergunta: {query}"
    resp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role':'system','content': 'Você é um assistente especializado.'},
                  {'role':'user','content': prompt}],
        max_tokens=512,
        temperature=0.2
    )
    return resp['choices'][0]['message']['content'].strip()

# --- Handlers do Telegram Bot ---

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Olá! Sou o bot da sua universidade. Pergunte o que quiser 😊")


def handle_message(update: Update, context: CallbackContext):
    user_q = update.message.text
    answer = generate_answer(user_q)
    update.message.reply_text(answer)


def main():
    # Inicializar DB
    init_db()
    # Processar PDF e armazenar embeddings (execute uma vez)
    PDF_PATH = 'info_universidade.pdf'
    chunks = extract_and_chunk_pdf(PDF_PATH)
    store_chunks(chunks)

    # Iniciar Telegram
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    print("Bot iniciado...")
    updater.idle()

if __name__ == '__main__':
    main()
