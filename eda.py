import streamlit as st
from PIL import Image
import pandas as pd

def run_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")

    # --- Imágenes del análisis (una por una) ---

    st.header("1. Tweet Map")
    st.image("images/world_map.png", caption="Geopraphic distribution of tweets")

    st.header("2. Most Frequent words")
    st.image("images/palabrasFrecuentes.png", caption="Words with low hate score")

    st.image("images/palabrasFrecuentesAlto.png", caption="Words with high hate score")

    st.header("3. Hate Speech Distribution")
    st.image("images/distribution_hateSpeech.png", caption="Class distribution")

    st.header("4. Hate Speech Density")
    st.image("images/densidad_hateSpeech.png", caption="Density distribution")

    st.header("5. Tweet length")
    st.image("images/dist_long_tweets.png", caption="Tweet length distribution")

    # --- DataFrames o tablas ---
    st.markdown("---")
    st.header("📋 Data statistics")

    # Ejemplo de tablas (sustituye por los tuyos si ya los tienes)
    # Tabla 1
    st.subheader("Class Count")
    data = {
        "Clase": ["No Hate", "Hate"],
        "Cantidad": [58712, 9111]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Tabla 2
    st.subheader("Tweet length statistics")
    data2 = {
        "Estadística": ["Media", "Mediana", "Máximo", "Mínimo"],
        "Valor": [89.5, 78, 280, 5]
    }
    st.dataframe(pd.DataFrame(data2))
