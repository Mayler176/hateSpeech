def run_eda():

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    import nltk
    from nltk.corpus import stopwords

    # Descargar stopwords (si no está hecho)
    nltk.download('stopwords')

    # Cargar datos
    @st.cache_data
    def load_data():
        df = pd.read_csv("hateSpeech/train.csv")
        df = df[["class", "tweet"]]
        return df

    df = load_data()

    # Definir stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(["the", "and", "a", "of", "to", "is", "in", "that", "it", "on", "for", "with", "as", "this", "was", "but", "rt"])

    # Función para limpiar texto
    def clean_tweet(tweet):
        return ' '.join([word for word in tweet.split() if word.lower() not in stop_words])

    # Limpiar tweets
    df["tweet"] = df["tweet"].astype(str).apply(clean_tweet)

    # Token length y palabras únicas
    df["token_length"] = df["tweet"].apply(lambda x: len(x.split()))
    df["unique_words"] = df["tweet"].apply(lambda x: len(set(x.split())))

    # Página Streamlit
    st.title("📊 Análisis Exploratorio de Tweets - Hate Speech Dataset")

    # Mostrar dataframe limpio
    if st.checkbox("Mostrar dataset limpio"):
        st.dataframe(df.sample(10))

    # Distribución de clases
    st.subheader("Distribución de Clases")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='class', data=df, ax=ax1)
    st.pyplot(fig1)

    # Distribución de longitud de tweet
    st.subheader("Distribución de longitud de tweet por clase")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='class', y='token_length', data=df, ax=ax2)
    st.pyplot(fig2)

    # Palabras únicas por tweet
    st.subheader("Palabras únicas por tweet por clase")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='class', y='unique_words', data=df, ax=ax3)
    st.pyplot(fig3)

    # Wordcloud general
    st.subheader("Nube de palabras general")
    text = " ".join(df["tweet"])
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(text)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis('off')
    st.pyplot(fig4)

    # Wordcloud por clase
    st.subheader("Nubes de palabras por clase")
    for c in sorted(df["class"].unique()):
        st.markdown(f"**Clase {c}**")
        text_c = " ".join(df[df["class"] == c]["tweet"])
        wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(text_c)
        fig_c, ax_c = plt.subplots(figsize=(10, 5))
        ax_c.imshow(wc, interpolation='bilinear')
        ax_c.axis('off')
        st.pyplot(fig_c)

    # Tweets ruidosos
    st.subheader("Ejemplos de tweets cortos/ruidosos (menos de 4 palabras)")
    st.dataframe(df[df["token_length"] < 4][["tweet", "class"]].head(10))
