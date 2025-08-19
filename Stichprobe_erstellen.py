import pandas as pd
import re
import nltk
import spacy
import random


nltk.download('stopwords')
from nltk.corpus import stopwords

# Lade spaCy-Modell nur für Englisch
nlp_en = spacy.load("en_core_web_sm")
stopwords_en = set(stopwords.words('english'))

# Bereinigen + Lemmatisieren (nur Englisch)
def clean_and_lemmatize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # URLs entfernen
    text = re.sub(r"[^\w\s]", "", text)         # Sonderzeichen entfernen
    text = re.sub(r"\d+", "", text)             # Zahlen entfernen

    doc = nlp_en(text)
    tokens = [token.lemma_ for token in doc if token.text not in stopwords_en]

    return " ".join(tokens)

# Reddit-Daten laden (hier: Excel mit 'comment_body'-Spalte)
df = pd.read_excel('Reddit_korpus_translated.xlsx', header=0)
print(df.columns)

# Bereinigen pro Zeile (ohne Sprache erkennen)
df["cleaned_text"] = df["comment_body"].apply(clean_and_lemmatize)

# Exportieren (alles Englisch, keine Trennung)
df.to_excel("reddit_en_cleaned.xlsx", index=False)
print(f"Bereinigung Reddit abgeschlossen. Beiträge: {len(df)}")

# Steam-Daten laden (hier: Excel mit 'review'-Spalte)
dfSteam = pd.read_excel("Steam-Reviews_KAFKA.xlsx")
print(dfSteam.columns)

# Bereinigen pro Zeile (ohne Sprache erkennen)
dfSteam["cleaned_text"] = dfSteam["review"].apply(clean_and_lemmatize)

# Exportieren (alles Englisch)
dfSteam.to_excel("steam_en_cleaned.xlsx", index=False)
print(f"Bereinigung Steam abgeschlossen. Beiträge: {len(dfSteam)}")

# Steam und Reddit vorbereiten
steam_texts = dfSteam["cleaned_text"].dropna().tolist()
reddit_texts = df["cleaned_text"].dropna().tolist()

# Funktion, um kafkaeske Kontexte zu extrahieren
def extract_kafka_sentences_random(texts, n=30):
    results = []
    for text in texts:
        sentences = text.split('.')
        for sentence in sentences:
            if "kafkaesque" in sentence:
                results.append(sentence.strip())
    if len(results) <= n:
        return results
    else:
        random.seed(42)
        return random.sample(results, n)

# Extrahieren (zufällige Auswahl)
steam_kafka_sentences = extract_kafka_sentences_random(steam_texts)
reddit_kafka_sentences = extract_kafka_sentences_random(reddit_texts)

# Ausgabe der zufällig ausgewählten Sätze
print("REDDIT – Zufällige kafkaesque Sätze:")
for i, sentence in enumerate(reddit_kafka_sentences):
    print(f"{i+1}. {sentence}")

print("\nSTEAM – Zufällige kafkaesque Sätze:")
for i, sentence in enumerate(steam_kafka_sentences):
    print(f"{i+1}. {sentence}")

    

#In Excel speichern
df_reddit_original = pd.DataFrame({"kafka_sentence": reddit_original_sentences})
df_steam_original = pd.DataFrame({"kafka_sentence": steam_original_sentences})

df_reddit_original.to_excel("Reddit_Kategorisierungen_Stichprobenauswahl.xlsx", index=False)
df_steam_original.to_excel("Steam_Kategorisierungen_Stichprobenauswahl.xlsx", index=False)

print("gespeichert")