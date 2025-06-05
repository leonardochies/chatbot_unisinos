import os
import pickle
import dotenv
import numpy as np
import faiss
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from mock_db import mock_alunos

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

client = OpenAI()

# Carrega dados e índice FAISS
with open("docs.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("faiss_index.index")

import spacy

nlp = spacy.load("pt_core_news_sm")

def detectar_intencao_dados_pessoais(texto):
    doc = nlp(texto.lower())

    temas_pessoais = {
        "notas": ["nota", "notas", "boletim", "tirei", "pontuação", "desempenho"],
        "horario": ["horário", "horario", "aulas", "matérias", "disciplinas", "agenda"],
        "historico": ["histórico", "historico", "disciplinas cursadas", "cadeiras", "aprovadas", "reprovadas"]
    }

    for token in doc:
        for tipo, palavras in temas_pessoais.items():
            if token.lemma_ in palavras:
                return tipo  # Ex: "notas", "horario", "historico"

    return None  # Nenhuma intenção clara

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

def buscar_dados_aluno(user_id):
    for aluno in mock_alunos:
        if aluno["id"] == user_id:
            return aluno
    return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Olá! Sou o chatbot da Unisinos. Me pergunte algo!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.lower().strip()

    intencao = detectar_intencao_dados_pessoais(user_text)
    requer_dados_pessoais = intencao is not None

    # Se o bot estiver aguardando Matrícula
    if context.user_data.get("esperando_id"):
        context.user_data["matricula_aluno"] = user_text
        context.user_data.pop("esperando_id")
        await update.message.reply_text("Número da matrícula salvo! Agora me pergunte novamente sua dúvida.")
        return

    # Se precisar de dados pessoais e ainda não temos a matrícula
    if requer_dados_pessoais and "matricula_aluno" not in context.user_data:
        await update.message.reply_text("Por favor, informe o número da sua matrícula para que eu possa acessar suas informações.")
        context.user_data["esperando_id"] = True
        return

    # Se tivermos uma matrícula, buscar dados do aluno
    aluno = None
    if "matricula_aluno" in context.user_data:
        aluno = buscar_dados_aluno(context.user_data["matricula_aluno"])

    # Contexto geral + contexto do aluno (se aplicável)
    try:
        await update.message.reply_text("Pensando... 🤖")

        context_chunks = search_similar_chunks(user_text)
        contexto_geral = "\n".join(context_chunks)

        contexto_aluno = ""
        if aluno:
            contexto_aluno = (
                f"Dados do aluno:\n"
                f"Nome: {aluno['nome']}\n"
                f"Notas: {aluno['notas']}\n"
                f"Horário: {aluno['horario']}\n"
                f"Intenção detectada: {intencao}\n"
            )

        prompt_inicial = (
            "Você é um assistente virtual da Universidade Unisinos. "
            "Se forem feitas perguntas pessoais como notas, horários ou histórico acadêmico, use os dados fornecidos do aluno para responder e caso não tenha nenhum dado do aluno, informe que os dados do aluno não foram encontrados e que é necessário chamar suporte humano."
            "Se a pergunta for geral, responda com base nas informações do contexto geral e caso não tenha contexto sobre a pergunta do usuário, informe que você não consegue ajudar e que é necessário chamar suporte humano."
        )

        messages = [
            {"role": "system", "content": prompt_inicial},
            {"role": "user", "content": f"{contexto_aluno}\n\nContexto geral:\n{contexto_geral}\n\nPergunta: {user_text}"}
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
