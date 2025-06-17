def run_tuning():


    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import os


    # --- Configurar la página ---
    st.title("🚀 Hyperparameter Tuning Dashboard")
    st.markdown("""
        In this page you will find the results of hyperparameter tuning for each model used.
    """)

    # --- Cargar datos ---
    def load_data():
        df_roberta = pd.read_csv("./tuning_results_roberta.csv")
        df_deberta = pd.read_csv("./tuning_results_deberta.csv")
        df_distilbert = pd.read_csv("./tuning_results_distilbert.csv")

        return df_roberta, df_deberta, df_distilbert

    # --- Datos ---
    df_roberta, df_deberta, df_distilbert = load_data()

    # --- Selector de modelo ---
    model_choice = st.selectbox("Selecciona el modelo:", ["RoBERTa", "DeBERTa", "DistilBERT"])
    df_selected = df_roberta if model_choice == "RoBERTa" else df_deberta if model_choice == "DeBERTa" else df_distilbert

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
    st.subheader("📋 Results")
    st.dataframe(df_selected.sort_values(by="f1_score", ascending=False).reset_index(drop=True), use_container_width=True)


    def run_tuning_explanation():
        st.subheader("🪶 Justification")
        st.markdown("""
    This section describes the parameters we tuned for our hate speech classification model based on **DistilBERT** and the rationale behind each choice.
    """)

        with st.expander("🔧 **Which parameters were tuned and why?**"):
            st.markdown("""
    We optimized the following **three hyperparameters** using a grid search to improve the model's performance on hate speech detection:

    ### 1️⃣ `max_length`
    - **What it does:** Sets the maximum number of tokens per tweet.
    - **Values tested:** `64` and `128`.
    - **Why:** Shorter lengths speed up processing but may lose context; longer lengths preserve important semantic cues which are crucial for detecting subtle hate speech.

    ### 2️⃣ `batch_size`
    - **What it does:** Determines how many samples are processed simultaneously.
    - **Values tested:** `16` and `32`.
    - **Why:** Larger batch sizes are faster on GPUs, while smaller ones help with memory limitations and prediction stability on variable-length text.

    ### 3️⃣ `threshold`
    - **What it does:** Converts the predicted probability of class "Hate" into binary labels.
    - **Values tested:** `0.3`, `0.4`, `0.5`, and `0.6`.
    - **Why:** Choosing the right threshold improves classification in imbalanced or ambiguous datasets. Using a fixed `0.5` threshold is often suboptimal for real-world data.
    """)

        with st.expander("📊 **How were they evaluated?**"):
            st.markdown("""
    Each combination of parameters was evaluated using the **F1-score**, which balances precision and recall.  
    This is especially important for hate speech detection, where both false positives and false negatives carry high risk.

    The best configuration was automatically selected and saved in a `.json` file for later use.  
    All tested combinations and their F1-scores were stored in a `.csv` file to support visualization and comparison across trials.
    """)

        with st.expander("📁 **Files generated**"):
            st.markdown("""
    - `**mejores_parametros_distilbert.json**`: Best hyperparameter configuration.
    - `**tuning_results_distilbert.csv**`: All evaluated combinations with corresponding F1-scores.
    """)

    run_tuning_explanation()