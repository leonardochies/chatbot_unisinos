import spacy

nlp = spacy.load("pt_core_news_sm")

def detect_intention(user_text):
    doc = nlp(user_text.lower())

    personal_themes = {
        "notas": ["nota", "notas", "boletim", "tirei", "pontuação", "desempenho"],
        "horario": ["horário", "horario", "aulas", "matérias", "disciplinas", "agenda"],
        "historico": ["histórico", "historico", "disciplinas cursadas", "cadeiras", "aprovadas", "reprovadas"]
    }

    for token in doc:
        for type, words in personal_themes.items():
            if token.lemma_ in words:
                return type

    return None