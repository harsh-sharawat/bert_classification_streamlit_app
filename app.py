# app.py

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(page_title="BERT Sentiment Analysis", layout="wide")

st.title("🧠 BERT-based Sentiment Analysis")
st.write("Enter a sentence and let BERT predict the sentiment (Positive / Negative).")

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Text input
user_input = st.text_area("Enter text:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs).item()

        labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        st.markdown(f"### Prediction: {labels[predicted_class]}")
        st.bar_chart(probs.squeeze().tolist())
