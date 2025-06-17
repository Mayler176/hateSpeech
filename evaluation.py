import streamlit as st
from PIL import Image

def run_eva():
    st.title("📈 Evaluación de Modelos - Hate Speech Detection")

    st.header("1. Matrices de Confusión")
    st.subheader("DeBERTa")
    st.image(Image.open("roberta_confusionMatrix.png"), caption="Matriz de Confusión - DeBERTa")

    st.subheader("DistilBERT")
    st.image(Image.open("distilbert_confusionMatrix.png"), caption="Matriz de Confusión - DistilBERT")

    st.subheader("Modelo con Oversampling")
    st.image(Image.open("descarga.png"), caption="Matriz de Confusión - Oversampling")

    st.markdown("---")
    st.header("2. Model Justification (10 pts)")

    st.markdown("""
We selected three state-of-the-art transformer-based models for hate speech detection: **DistilBERT**, 
**RoBERTa**, and **DeBERTa**. These models were chosen due to their demonstrated effectiveness in
natural language understanding tasks, especially those requiring nuanced language interpretation like
hate speech detection. The architecture of each model offers unique benefits that align with the needs
of our problem, ensuring robust performance across a range of linguistic inputs.

**DistilBERT** is a distilled version of BERT, significantly smaller and faster, yet still powerful.  
It retains approximately 97% of BERT’s performance while using 40% fewer parameters, making it  
optimal for real-time applications and limited-resource environments. Its ability to generalize  
well despite being lightweight makes it an excellent baseline for benchmarking more complex models.

**RoBERTa** enhances the BERT architecture by removing the next sentence prediction objective  
and employing dynamic masking and longer training. These changes make it more robust for  
capturing intricate semantic relationships, which is critical in hate speech where hostility may be  
subtly implied rather than explicitly stated. Its consistent top-tier performance across NLP  
benchmarks supports its selection.

**DeBERTa** introduces a novel disentangled attention mechanism and enhanced positional  
embeddings, allowing the model to treat content and position separately. This makes it  
especially effective in understanding contextual and indirect language—a common trait in hate  
speech. Additionally, its improved generalization capabilities make it suitable for diverse and  
imbalanced datasets.
""")

    st.markdown("---")
    st.header("3. Classification Report (10 pts)")

    st.subheader("DistilBERT")
    st.code("""
              precision    recall  f1-score   support

    No Hate       0.83      0.86      0.85       890
       Hate       0.84      0.81      0.82       889

    accuracy                           0.83      1779
   macro avg       0.83      0.83      0.83      1779
weighted avg       0.83      0.83      0.83      1779
    """)

    st.subheader("RoBERTa")
    st.code("""
              precision    recall  f1-score   support

    No Hate       0.82      0.88      0.85       890
       Hate       0.86      0.79      0.82       889

    accuracy                           0.84      1779
   macro avg       0.84      0.84      0.84      1779
weighted avg       0.84      0.84      0.84      1779
    """)

    st.subheader("DeBERTa")
    st.code("""
              precision    recall  f1-score   support

    No Hate       0.87      0.85      0.86       890
       Hate       0.85      0.87      0.86       889

    accuracy                           0.86      1779
   macro avg       0.86      0.86      0.86      1779
weighted avg       0.86      0.86      0.86      1779
    """)

    st.markdown("---")
    st.header("4. Confusion Matrix (5 pts)")
    st.markdown("""
Each model was evaluated using a labeled heatmap of the confusion matrix. The axes are clearly  
marked with "Predicted" and "True", and class labels "No Hate" and "Hate" are displayed. From the  
heatmaps:

- **RoBERTa** demonstrates high recall for the "No Hate" class but underperforms slightly in  
  identifying "Hate".

- **DistilBERT** shows balanced classification performance across both labels.

- **DeBERTa** achieves the most consistent accuracy, showing fewer misclassifications in both  
  directions.

These matrices offer visual confirmation of performance trends observed in the classification reports.
""")

    st.markdown("---")
    st.header("5. Error Analysis (10 pts)")
    st.markdown("""
Misclassifications across models tend to occur in linguistically ambiguous or culturally contextual  
expressions. For instance:

- A sentence like *"Sure, they deserve it..."* was tagged as "No Hate" by DistilBERT despite having a  
  sarcastic and hostile undertone.

- The phrase *"Go back to where you came from"* was misclassified by RoBERTa, likely due to the  
  lack of overt aggression.

Such examples highlight the models’ difficulties in detecting sarcasm, euphemisms, and culturally  
loaded phrases. These errors indicate limitations in capturing pragmatic meaning without external  
context.

To improve performance, we suggest the following enhancements:

- Incorporate domain-specific corpora with varied hate speech patterns and cultural expressions  
  for fine-tuning.

- Augment input features with linguistic indicators like sentiment scores or part-of-speech tags.

- Integrate contextual signals from metadata (e.g., user history or hashtags) or ensemble  
  approaches using complementary models.

These strategies could boost the models’ ability to recognize implicit and context-heavy hate speech  
more reliably.
""")
