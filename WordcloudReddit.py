from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import re
import nltk
import spacy
from langdetect import detect, LangDetectException
from wordcloud import STOPWORDS

try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')  # Testzugriff
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords


# Lade spaCy-Modell nur für Englisch
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp_en = spacy.load("en_core_web_sm")

# Lade spaCy-Modell mit try, damit Ausführung schneller
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp_en = spacy.load("en_core_web_sm")

# Eigene Stopwords
custom_stopwords = STOPWORDS.union(set([
    "game", "find", "play", "one", "give", "contain", "way", "even", "say", "bit",
    "buy", "try", "seem", "make", "go", "word", "far", "truly", "despite", "think",
    "thing", "absolutely", "something", "nothing", "well", "back", "set", "man",
    "quite", "become", "though", "lot", "take", "begin", "anyone", "sometimes", "much", "almost", "pretty", "might"
]))

stopwords_en = set(stopwords.words('english'))

# Bereinigen + Lemmatisieren
def clean_and_lemmatize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # URLs 
    text = re.sub(r"[^\w\s]", "", text)         # Sonderzeichen 
    text = re.sub(r"\d+", "", text)             # Zahlen 

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

# Reddit-Daten laden 
df = pd.read_excel("Reddit_korpus_translated.xlsx")
print(df.columns)

# Bereinigen pro Zeile (ohne Sprache erkennen)
df["cleaned_text"] = df["Satz"].apply(clean_and_lemmatize)

# Schritt 1: Texte aus beiden Korpora zusammenführen
texts = df["cleaned_text"].dropna().tolist()

# Schritt 2: Kontextwörter von 'kafkaesk' erfassen
window_size = 3
kafkaesk_context_words = []

for text in texts:
    words = text.split()
    for i, word in enumerate(words):
        if word in ['kafkaesque', 'kafkaesk']:
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(words))
            context_window = words[start:end]
            filtered_context_words = [w for w in context_window if w not in custom_stopwords]
            kafkaesk_context_words.extend(filtered_context_words)

# Häufigkeiten berechnen
freq_kafkaesk = Counter(kafkaesk_context_words)

print("Top 15 Wörter im Kontext von 'kafkaesk':")
print(freq_kafkaesk.most_common(20))

text_for_wordcloud = ' '.join(kafkaesk_context_words)

wordcloud = WordCloud(width=1000, height=500, background_color='white', max_words=20, stopwords=custom_stopwords).generate(text_for_wordcloud)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Reddit: Context of 'kafkaesque'", fontsize=18)
plt.show()