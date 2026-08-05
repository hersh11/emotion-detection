from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = Path("artifacts")
LABEL_MAP_PATH = Path("label_map.json")
THRESHOLD = 0.25


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


@st.cache_data
def load_labels():
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
        label_map = json.load(file)
    return {int(key): value for key, value in label_map.items()}


def predict(text, tokenizer, model, id_to_label):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits).numpy()[0]

    selected = [
        (id_to_label[index], float(probability))
        for index, probability in enumerate(probabilities)
        if probability > THRESHOLD
    ]
    selected.sort(key=lambda item: item[1], reverse=True)

    return selected, probabilities


st.set_page_config(page_title="Emotion Detection", page_icon="ED", layout="wide")

st.title("Emotion Detection")
st.caption("Multi-label emotion detection trained with a DeBERTa v3 model on GoEmotions.")

if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
    st.error("Model artifacts or label_map.json were not found. Add them before running the app.")
    st.stop()

tokenizer, model = load_model()
id_to_label = load_labels()

text = st.text_area(
    "Enter text",
    height=130,
    placeholder="Example: I am happy and grateful today.",
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        emotions, all_probabilities = predict(text, tokenizer, model, id_to_label)

        st.subheader("Detected emotions")

        if not emotions:
            st.write("No emotion crossed the confidence threshold.")
        else:
            for label, probability in emotions:
                st.write(f"**{label.capitalize()}** `{probability:.2f}`")

        chart_data = pd.DataFrame(
            {
                "emotion": [id_to_label[index] for index in range(len(all_probabilities))],
                "probability": all_probabilities,
            }
        )

        chart = (
            alt.Chart(chart_data.sort_values(by="probability", ascending=False))
            .mark_bar(color="#6d5dfc")
            .encode(
                x=alt.X("probability:Q", title="Confidence score"),
                y=alt.Y("emotion:N", sort="-x", title="Emotion"),
                tooltip=["emotion", "probability"],
            )
            .properties(title="Prediction confidence")
        )

        st.subheader("Probability distribution")
        st.altair_chart(chart, use_container_width=True)

st.caption("Built with Streamlit and Hugging Face Transformers.")
