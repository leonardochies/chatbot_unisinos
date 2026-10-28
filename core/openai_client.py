import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def generate_response(user_text, context_chunks, student=None, intention=None):
    general_context = "\n".join(context_chunks)

    student_context = ""
    if student:
        student_context = (
            f"Dados do aluno:\n"
            f"Nome: {student['nome']}\n"
            f"Notas: {student['notas']}\n"
            f"Horário: {student['horario']}\n"
            f"Intenção detectada: {intention}\n"
        )

    initial_prompt = (
        "Você é um assistente virtual da Universidade Unisinos. "
        "Se forem feitas perguntas pessoais como notas, horários ou histórico acadêmico, use os dados fornecidos do aluno para responder e caso não tenha nenhum dado do aluno, informe que os dados do aluno não foram encontrados e que é necessário chamar suporte humano. "
        "Se a pergunta for geral, responda com base nas informações do contexto geral e caso não tenha contexto sobre a pergunta do usuário, informe que você não consegue ajudar e que é necessário chamar suporte humano."
    )

    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": f"{student_context}\n\nContexto geral:\n{general_context}\n\nPergunta: {user_text}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.strip()