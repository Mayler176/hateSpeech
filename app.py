import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# --- Configuración (ajusta según tus hyperparameter tuning) ---
MODEL_NAME = "AnaPau777/distibertHate"
THRESHOLD = 0.4
MAX_LENGTH = 128

# --- Cargar modelo y tokenizer (con cache para acelerar carga) ---
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# --- Predicción de hate speech ---
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()
        probs = softmax(logits, axis=1)[0]
        hate_score = probs[1]  # Clase 1 = Hate
        label = "HATE SPEECH" if hate_score > THRESHOLD else "NO HATE"
        return label, hate_score

# --- Interfaz Streamlit ---
st.title("🔍 Detección de Hate Speech con DistilBERT")
st.markdown("Modelo cargado desde: `AnaPau777/distibertHate`")

user_input = st.text_area("Escribe un texto para analizar:")

if st.button("Analizar"):
    if user_input.strip():
        label, score = predict(user_input)
        st.markdown(f"**Resultado:** `{label}`")
        st.markdown(f"**Score de hate speech:** `{score:.4f}`")
    else:
        st.warning("Por favor escribe algo para analizar.")
