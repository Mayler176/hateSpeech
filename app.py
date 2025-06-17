import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Asegura uso de CPU en Streamlit Cloud

import streamlit as st

# Menú de navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["Predicción", "Análisis EDA", "Evaluation and Justification"])

if page == "Predicción":
    from prediction import run_prediction
    run_prediction()

elif page == "Análisis EDA":
    from eda import run_eda
    run_eda()

elif page == "Evaluation and Justification":
    from evaluation import run_eva
    run_eva()
