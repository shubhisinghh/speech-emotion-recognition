import streamlit as st
import joblib
import tempfile
import os

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("Speech Emotion Recognition")
st.write("Upload a WAV audio file and the model will predict the emotion.")

@st.cache_resource
def load_model():
    return joblib.load("emotion_model.pkl")

rec = load_model()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    with st.spinner("Predicting emotion..."):
        prediction = rec.predict(audio_path)
        probabilities = rec.predict_proba(audio_path)

    st.success(f"Predicted emotion: {prediction}")

    st.subheader("Emotion Probabilities")
    for emotion, prob in probabilities.items():
        st.write(f"{emotion}: {prob*100:.2f}%")
        st.progress(float(prob))
