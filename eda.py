import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset

nltk.download('stopwords')

def run_eda():
    st.title("Exploración de Datos - Hate Speech")

    # --- Cargar datos desde Hugging Face ---
    dataset = load_dataset("AnaPau777/hateSpeech", split="train")
    data = pd.DataFrame(dataset)

    # Renombrar columnas
    data = data.rename(columns={"text": "text", "hate speech score": "hate_speech_score"})
    data = data[["text", "hate_speech_score"]].dropna()

    # Definir stopwords
    stop_words = set(stopwords.words('english'))
    new_stopwords = ["the", "and", "a", "of", "to", "is", "in", "that", "it", "on", "for", "with", "as", "this", "was", "but", "rt", "url"]
    stop_words.update(new_stopwords)

    # Limpiar texto
    def clean_tweet(tweet):
        return ' '.join([word for word in tweet.split() if word.lower() not in stop_words])
    
    data["text"] = data["text"].astype(str).apply(clean_tweet)

    # Métricas básicas
    data['token_length'] = data["text"].apply(lambda x: len(x.split()))
    data['unique_words'] = data["text"].apply(lambda x: len(set(x.lower().split())))

    # WordCloud
    st.subheader("Nube de Palabras General")
    text = " ".join(data['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Histograma hate speech score
    st.subheader("Distribución del Hate Speech Score")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data['hate_speech_score'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Longitud de tweets
    st.subheader("Distribución de Longitud de Tweets")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data['token_length'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # KDE 2D
    st.subheader("Densidad: Hate Speech Score vs Palabras Únicas")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=data, x='hate_speech_score', y='unique_words', cmap="Reds", fill=True, thresh=0.05, ax=ax)
    st.pyplot(fig)

    # Tweets extremos
    st.subheader("Tweets con Hate Speech Score Más Alto")
    st.dataframe(data.sort_values('hate_speech_score', ascending=False).head(5)[['text', 'hate_speech_score']])

    st.subheader("Tweets con Hate Speech Score Más Bajo")
    st.dataframe(data.sort_values('hate_speech_score', ascending=True).head(5)[['text', 'hate_speech_score']])

    # Palabras frecuentes
    def plot_top_words(text, title):
        words = [w for w in text.split() if w not in stop_words]
        counter = Counter(words)
        common = counter.most_common(15)
        if common:
            words, freqs = zip(*common)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=list(freqs), y=list(words), ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
        else:
            st.warning(f"No se encontraron palabras frecuentes para: {title}")

    low_score_text = " ".join(data[data['hate_speech_score'] < 0.25]['text'])
    high_score_text = " ".join(data[data['hate_speech_score'] > 0.75]['text'])

    st.subheader("Palabras frecuentes en Tweets con bajo hate score")
    plot_top_words(low_score_text, "Palabras más frecuentes en Tweets con bajo hate score")

    st.subheader("Palabras frecuentes en Tweets con alto hate score")
    plot_top_words(high_score_text, "Palabras más frecuentes en Tweets con alto hate score")
