# 🤖 Chatbot Unisinos com OpenAI e Telegram

Este projeto é um chatbot integrado ao Telegram que responde perguntas gerais e pessoais (como notas e horários) de alunos da Unisinos. Ele utiliza embeddings, busca semântica com FAISS, e completions da OpenAI.

---

## 📦 Estrutura do Projeto

├── main.py
├── bot/
│ ├── telegram_bot.py
│ ├── handlers.py
│ └── user_context.py
├── core/
│ ├── intent.py
│ ├── search.py
│ ├── openai_client.py
│ └── embedding_utils.py
├── data/
│ ├── loader.py
│ ├── pdf_utils.py
│ └── mock_db.py
├── scripts/
│ └── generate_embeddings.py
├── docs.pkl
├── faiss_index.index
├── requirements.txt

## ⚙️ Como usar

### 1. Crie e ative o ambiente virtual

```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2. Instale as dependências

```
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### 3. Configure as variáveis de ambiente

Por conta do ambiente virtual, é necessário setar diretamente as variáveis antes de rodar os scripts:

## Linux/macOS:

```
export OPENAI_API_KEY=your_openai_key
export TELEGRAM_TOKEN=your_telegram_bot_token
```

## Windows (CMD):

```
set OPENAI_API_KEY=your_openai_key
set TELEGRAM_TOKEN=your_telegram_bot_token
```

## Windows (PowerShell):

```
$env:OPENAI_API_KEY="your_openai_key"
$env:TELEGRAM_TOKEN="your_telegram_bot_token"
```

### 4. Gere os embeddings (apenas uma vez ou quando o FAQ mudar)

```
python scripts/generate_embeddings.py
```

### 5. Inicie o bot

```
python main.py
```