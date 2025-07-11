def run_prediction():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Asegura uso de CPU en Streamlit Cloud

    import streamlit as st
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # --- Configuración global ---
    MODELOS = {
        "DistilBERT": {
            "name": "AnaPau777/distibertHate",
            "threshold": 0.4,
            "max_length": 128
        },
        "RoBERTa": {
            "name": "AnaPau777/roberta",
            "threshold": 0.3,
            "max_length": 128
        },
        "DeBERTa": {
            "name": "Narrativaai/deberta-v3-small-finetuned-hate_speech18",
            "threshold": 0.4,
            "max_length": 128
        }
    }

    # --- Cargar modelos/tokenizers con cache ---
    @st.cache_resource
    def load_model_and_tokenizer(model_name, model_key):
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # DeBERTa requiere use_fast=False para evitar tokenizer.json corrupto
        if model_key == "DeBERTa":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.eval()
        return model, tokenizer

    # --- Predicción con el modelo seleccionado ---
    def predict(text, model, tokenizer, max_length, threshold):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0]
            hate_score = probs[1].item()
            label = "HATE SPEECH" if hate_score > threshold else "NO HATE"
            return label, hate_score

    # --- Interfaz de usuario Streamlit ---
    st.title("🔍 Hate Speech Detector")
    url = "https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech"

    st.markdown('''
                :orange-badge[⚠️ Hey!]
    :rainbow[*note that these models were trained using tweets. The input should resemble this type of content to ensure the best posible results. See some examples* [here](%s) ] '''% url)
    st.markdown("**Select a model to analyze the text:**")

    modelo_elegido = st.selectbox("Modelo", list(MODELOS.keys()))
    modelo_config = MODELOS[modelo_elegido]

    model, tokenizer = load_model_and_tokenizer(modelo_config["name"], modelo_elegido)

    user_input = st.text_area("Write a hate related type of tweet")

    if st.button("Analizar"):
        if user_input.strip():
            label, score = predict(
                text=user_input,
                model=model,
                tokenizer=tokenizer,
                max_length=modelo_config["max_length"],
                threshold=modelo_config["threshold"]
            )
            st.markdown(f"**Modelo:** `{modelo_elegido}`")
            st.markdown(f"**Resultado:** `{label}`")
            st.markdown(f"**Score de hate speech:** `{score:.4f}`")
        else:
            st.warning("Please, write a tweet to analyze it")
