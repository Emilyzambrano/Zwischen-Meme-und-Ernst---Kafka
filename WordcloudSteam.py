from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import re
import nltk
import spacy
import random
from langdetect import detect, LangDetectException
from wordcloud import STOPWORDS

# manuell gewählte Stopwords
custom_stopwords = STOPWORDS.union(set(["game", "find", "play", "one", "give", "contain","way", "even", "say", "bit", "buy", "try", "seem", "find", "make", "go", "give", "word", "far", "truly", "despite", "think", "thing", "absolutely", "something", "nothing", "well", "back", "set", "man", "quite", "become", "though", "lot", "take", "begin"]))


nltk.download('stopwords')
from nltk.corpus import stopwords

# Lade spaCy-Modell (Englisch)
nlp_en = spacy.load("en_core_web_sm")
stopwords_en = set(stopwords.words('english'))

# Bereinigen + Lemmatisieren 
    
def clean_and_lemmatize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # URLs entfernen
    text = re.sub(r"[^\w\s]", "", text)         # Sonderzeichen entfernen
    text = re.sub(r"\d+", "", text)             # Zahlen entfernen

    doc = nlp_en(text)
    tokens = [token.lemma_ for token in doc if token.text not in stopwords_en]

    return " ".join(tokens)

# Funktion zur Sprachenerkennung mit Fehlerbehandlung
def detect_language(text):
    if not isinstance(text, str) or text.strip() == "":
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# Steam-Daten laden (hier: Excel mit 'review'-Spalte)
dfSteam = pd.read_excel("Steam-Reviews_KAFKA.xlsx")
print(dfSteam.columns)

# Sprache erkennen
dfSteam['detected_language'] = dfSteam['review'].apply(detect_language)

# Nur englische Reviews auswählen
dfSteam_en = dfSteam[dfSteam['detected_language'] == 'en'].copy()
print(f"Anzahl englischer Reviews: {len(dfSteam_en)}")

# Bereinigen pro Zeile
dfSteam_en["cleaned_text"] = dfSteam_en["review"].apply(clean_and_lemmatize)

# Schritt 1: 
texts = dfSteam_en["cleaned_text"].dropna().tolist()

# Schritt 2: Kontextwörter von 'kafkaesk' erfassen
window_size = 3
kafkaesk_context_words = []

for text in texts:
    words = text.split()
    for i, word in enumerate(words):
        if word == 'kafkaesque':
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(words))
            kafkaesk_context_words.extend(words[start:end])

# Häufigkeiten berechnen
freq_kafkaesk = Counter(kafkaesk_context_words)

print("Top 15 Wörter im Kontext von 'kafkaesk':")
print(freq_kafkaesk.most_common(15))

text_for_wordcloud = ' '.join(kafkaesk_context_words)

wordcloud = WordCloud(width=1000, height=500, background_color='white', max_words=45, stopwords=custom_stopwords).generate(text_for_wordcloud)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Steam-Reviews: Context of 'kafkaesque'", fontsize=18)
plt.show()