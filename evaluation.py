import streamlit as st
from PIL import Image

def run_eva():
    st.title("🤖 Model Evaluation Dashboard")

    st.markdown("### Select the model you want to visualize:")
    model = st.selectbox("Available models:", ["RoBERTa", "DistilBERT", "DeBERTa"])

    # Mostrar imagen y clasificación según el modelo seleccionado
    st.header("📌 Confusion Matrix")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(f"### {model}")
        if model == "RoBERTa":
            img_path = "images/roberta_confusionMatrix.png"
        elif model == "DistilBERT":
            img_path = "images/distilbert_confusionMatrix.png"
        else:
            img_path = "images/deberta_confusionMatrix.png"
        st.image(Image.open(img_path), caption=f"Confusion Matrix - {model}")

    with col2:
        st.markdown("### Classification Report")
        if model == "DistilBERT":
            st.code("""
              precision    recall  f1-score   support

    No Hate       0.83      0.86      0.85       890
       Hate       0.84      0.81      0.82       889

    accuracy                           0.83      1779
   macro avg       0.83      0.83      0.83      1779
weighted avg       0.83      0.83      0.83      1779
            """)
        elif model == "RoBERTa":
            st.code("""
              precision    recall  f1-score   support

    No Hate       0.82      0.88      0.85       890
       Hate       0.86      0.79      0.82       889

    accuracy                           0.84      1779
   macro avg       0.84      0.84      0.84      1779
weighted avg       0.84      0.84      0.84      1779
            """)
        else:
            st.code("""
              precision    recall  f1-score   support

    No Hate       0.87      0.85      0.86       890
       Hate       0.85      0.87      0.86       889

    accuracy                           0.86      1779
   macro avg       0.86      0.86      0.86      1779
weighted avg       0.86      0.86      0.86      1779
            """)

    st.markdown("---")

    with st.expander("📚 Model Justification"):
        st.markdown("""
We selected three state-of-the-art transformer-based models for hate speech detection: **DistilBERT**, 
**RoBERTa**, and **DeBERTa**. These models were chosen due to their demonstrated effectiveness in
natural language understanding tasks, especially those requiring nuanced language interpretation like
hate speech detection. The architecture of each model offers unique benefits that align with the needs
of our problem, ensuring robust performance across a range of linguistic inputs.

**DistilBERT** is a distilled version of BERT, significantly smaller and faster, yet still powerful.  
It retains approximately 97% of BERT’s performance while using 40% fewer parameters, making it  
optimal for real-time applications and limited-resource environments.

**RoBERTa** enhances BERT by removing the next sentence prediction objective and applying  
dynamic masking and longer training, making it more robust for detecting implicit hostility.

**DeBERTa** introduces disentangled attention and improved positional embeddings, enhancing its  
ability to interpret contextual and indirect language effectively.
""")

    with st.expander("📊 Confusion Matrix Analysis"):
        st.markdown("""
- **RoBERTa**: High recall for "No Hate", struggles slightly with "Hate".
- **DistilBERT**: Balanced across classes.
- **DeBERTa**: Most consistent accuracy, fewer misclassifications.
""")

    with st.expander("🧠 Error Analysis"):
        st.markdown("""
- Misclassifications appear in sarcastic or culturally loaded texts.
- Examples:
    - *"Sure, they deserve it..."* → Misclassified as No Hate by DistilBERT.
    - *"Go back to where you came from."* → Misclassified by RoBERTa.

**Suggestions for Improvement**:
- Fine-tuning with domain-specific corpora.
- Adding sentiment/POS features.
- Incorporating context (hashtags, user history) or ensembles.
""")
