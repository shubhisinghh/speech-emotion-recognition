import streamlit as st
from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
import tempfile
import os

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("Speech Emotion Recognition")
st.write("Upload a WAV audio file and the model will predict the emotion.")

@st.cache_resource
def load_model():
    model = SVC(probability=True)
    rec = EmotionRecognizer(
        model=model,
        emotions=["sad", "neutral", "happy"],
        balance=True,
        verbose=0
    )
    rec.train()
    return rec

rec = load_model()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    with st.spinner("Predicting emotion..."):
        prediction = rec.predict(temp_audio_path)
        probabilities = rec.predict_proba(temp_audio_path)

    st.success(f"Predicted emotion: {prediction}")

    st.subheader("Emotion Probabilities")

    for emotion, prob in probabilities.items():
        st.write(f"{emotion}: {prob*100:.2f}%")
        st.progress(float(prob))

