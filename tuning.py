def run_tuning():


    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import os


    st.write("Current working directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())


    # --- Configurar la página ---
    st.title("🚀 Hyperparameter Tuning Dashboard")
    st.markdown("""
    Esta página interactiva muestra los resultados del ajuste de hiperparámetros
    para modelos *RoBERTa* y *DeBERTa*, utilizando estrategias como `Optuna` o `Keras Tuner`.

    Puedes analizar el impacto de combinaciones como `batch size`, `max length` y `threshold`
    sobre el F1-score, y revisar las mejores configuraciones obtenidas.
    """)

    # --- Cargar datos ---
    def load_data():
        df_roberta = pd.read_csv("./tuning_results_roberta.csv")
        df_deberta = pd.read_csv("./tuning_results_deberta.csv")

        return df_roberta, df_deberta

    # --- Datos ---
    df_roberta, df_deberta = load_data()

    # --- Selector de modelo ---
    model_choice = st.selectbox("Selecciona el modelo:", ["RoBERTa", "DeBERTa"])
    df_selected = df_roberta if model_choice == "RoBERTa" else df_deberta

    # --- Visualización interactiva ---
    st.subheader("📈 Evolución del F1-Score")
    fig = px.line(
        df_selected.reset_index(),
        x="index", y="f1_score",
        color=df_selected['batch_size'].astype(str),
        markers=True,
        labels={"index": "Trial", "f1_score": "F1 Score", "color": "Batch Size"},
        title=f"F1 Score por combinación de hiperparámetros - {model_choice}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Mostrar mejores hiperparámetros ---
    st.subheader("🏆 Mejor configuración encontrada")
    best_row = df_selected.loc[df_selected['f1_score'].idxmax()]
    st.markdown(f"""
    - **F1-score:** {best_row['f1_score']:.4f}  
    - **Batch size:** {best_row['batch_size']}  
    - **Max length:** {best_row['max_length']}  
    - **Threshold:** {best_row['threshold']}  
    """)

    # --- Tabla completa ---
    st.subheader("📋 Resultados completos")
    st.dataframe(df_selected.sort_values(by="f1_score", ascending=False).reset_index(drop=True), use_container_width=True)

    # --- Mostrar screenshots si existen ---
    screenshot_path = f"screenshots/{model_choice.lower()}"
    if os.path.exists(screenshot_path):
        st.subheader("🖼️ Capturas del proceso")
        image_files = [f for f in os.listdir(screenshot_path) if f.endswith(".png")]
        for img in sorted(image_files):
            st.image(f"{screenshot_path}/{img}", use_column_width=True)
    else:
        st.info("Agrega capturas del proceso en la carpeta 'screenshots/roberta' o 'screenshots/deberta' para mostrarlas aquí.")
