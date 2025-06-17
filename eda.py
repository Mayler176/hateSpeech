import streamlit as st
from PIL import Image
import pandas as pd

def run():
    st.title("📊 Análisis Exploratorio de Datos (EDA)")

    # --- Imágenes del análisis (una por una) ---

    st.header("1. Mapa Mundial de Tweets")
    st.image("world_map.png", caption="Distribución Geográfica de Tweets")

    st.header("2. Palabras Más Frecuentes")
    st.image("palabrasFrecuentes.png", caption="Palabras Frecuentes Generales")

    st.image("palabrasFrecuentesAlto.png", caption="Palabras Frecuentes en Tweets con Score Alto")

    st.header("3. Distribución de Hate Speech")
    st.image("distribution_hateSpeech.png", caption="Distribución de Clases")

    st.header("4. Densidad de Hate Speech")
    st.image("densidad_hateSpeech.png", caption="Distribución de Densidad")

    st.header("5. Longitud de Tweets")
    st.image("dist_long_tweets.png", caption="Distribución de Longitudes de Tweets")

    st.header("6. Matrices de Confusión (EDA)")
    st.image("roberta_confusionMatrix.png", caption="Confusión - RoBERTa")
    st.image("distilbert_confusionMatrix.png", caption="Confusión - DistilBERT")

    # --- DataFrames o tablas ---
    st.markdown("---")
    st.header("📋 Estadísticas de Datos (Tablas)")

    # Ejemplo de tablas (sustituye por los tuyos si ya los tienes)
    # Tabla 1
    st.subheader("Ejemplo: Conteo por Clases")
    data = {
        "Clase": ["No Hate", "Hate"],
        "Cantidad": [58712, 9111]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Tabla 2
    st.subheader("Ejemplo: Estadísticas de Longitud de Tweets")
    data2 = {
        "Estadística": ["Media", "Mediana", "Máximo", "Mínimo"],
        "Valor": [89.5, 78, 280, 5]
    }
    st.dataframe(pd.DataFrame(data2))
