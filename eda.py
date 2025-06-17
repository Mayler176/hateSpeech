import streamlit as st
from PIL import Image
import pandas as pd

def run_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("Welcome to the exploratory analysis of the dataset. The distributions, frequencies, and characteristics of tweets with and without hate speech are shown here.")

    # --- Tweet Map ---
    with st.expander("🗺️ 1. Tweets word map"):
        st.image("images/world_map.png", caption="Tweet word map", use_container_width=True)

    # --- Frequent Words ---
    with st.expander("🔤 2. Most frequent words"):
        col1, col2 = st.columns(2)
        with col1:
            st.image("images/palabrasFrecuentes.png", caption="Words in tweets with low hate score", use_container_width=True)
        with col2:
            st.image("images/palabrasFrecuentesAlto.png", caption="Words in tweets with high hate score", use_container_width=True)

    # --- Class Distribution ---
    with st.expander("📊 3. Class distribution"):
        st.image("images/distribution_hateSpeech.png", caption="Proportion of tweets with and without hate", use_container_width=True)

    # --- Hate Density ---
    with st.expander("🌡️ 4. Hate speech density"):
        st.image("images/densidad_hateSpeech.png", caption="Distribution of hate density", use_container_width=True)

    # --- Tweet Length ---
    with st.expander("✍️ 5. Tweet length"):
        st.image("images/dist_long_tweets.png", caption="Distribution of tweet length", use_container_width=True)

    # --- Tablas / Estadísticas ---
    st.markdown("---")
    st.header("📋 Dataset statistics")

    with st.expander("🔢 Class count"):
        data = {
            "Class": ["No Hate", "Hate"],
            "Count": [58712, 9111]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    with st.expander("📏 Length tweet statistics"):
        data2 = {
            "Statistic": ["Average", "Median", "Maximum", "Minimum"],
            "Value": [89.5, 78, 280, 5]
        }
        st.dataframe(pd.DataFrame(data2), use_container_width=True)
