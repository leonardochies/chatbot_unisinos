from telegram import Update
from telegram.ext import ContextTypes
from core.intent import detect_intention
from core.search import search_similar_chunks
from core.openai_client import generate_response
from bot.user_context import search_user_data

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Olá! Sou o chatbot da Unisinos. Me pergunte algo!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.lower().strip()
    intention = detect_intention(user_text)
    require_personal_data = intention is not None

    if context.user_data.get("waiting_id"):
        context.user_data["user_registration"] = user_text
        context.user_data.pop("waiting_id")
        await update.message.reply_text("Número da matrícula salvo! Agora me pergunte novamente sua dúvida.")
        return

    if require_personal_data and "user_registration" not in context.user_data:
        await update.message.reply_text("Por favor, informe o número da sua matrícula para que eu possa acessar suas informações.")
        context.user_data["waiting_id"] = True
        return

    student = None
    if "user_registration" in context.user_data:
        student = search_user_data(context.user_data["user_registration"])

    try:
        await update.message.reply_text("Pensando... 🤖")
        context_chunks = search_similar_chunks(user_text)
        response = generate_response(user_text, context_chunks, student, intention)
        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Erro ao processar sua pergunta: {str(e)}")