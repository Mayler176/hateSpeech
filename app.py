import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Asegura uso de CPU en Streamlit Cloud

import streamlit as st

# Menú de navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["Prediction", "EDA Analysis", "Evaluation and Justification", "Hyperparameter Tuning"])

if page == "Prediction":
    from prediction import run_prediction
    run_prediction()

elif page == "EDA Analysis":
    from eda import run_eda
    run_eda()

elif page == "Evaluation and Justification":
    from evaluation import run_eva
    run_eva()


elif page == "Hyperparameter Tuning":
    from tuning import run_tuning
    run_tuning()
