import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Fuerza uso de CPU en Streamlit Cloud

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuración del modelo ---
MODEL_NAME = "AnaPau777/distibertHate"
THRESHOLD = 0.4
MAX_LENGTH = 128

# --- Cargar modelo y tokenizer con cache ---
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
device = torch.device("cpu")

# --- Función de predicción ---
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        hate_score = probs[1].item()
        label = "HATE SPEECH" if hate_score > THRESHOLD else "NO HATE"
        return label, hate_score

# --- Interfaz de usuario Streamlit ---
st.title("🔍 Detección de Hate Speech - DistilBERT")
st.markdown("Modelo: `AnaPau777/distibertHate`")

user_input = st.text_area("Escribe un texto para analizar:")

if st.button("Analizar"):
    if user_input.strip():
        label, score = predict(user_input)
        st.markdown(f"**Resultado:** `{label}`")
        st.markdown(f"**Score:** `{score:.4f}`")
    else:
        st.warning("Por favor escribe algo para analizar.")
