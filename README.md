# Speech Emotion Recognition (Streamlit App)

A machine learning–based Speech Emotion Recognition web application that predicts human emotions from WAV audio files.
The app is built using Python, classical ML techniques, and deployed using Streamlit Cloud.

## Demo

Upload a WAV file and get emotion predictions instantly through an interactive web interface.

## Features

-Upload WAV audio files
-Extract audio features (MFCC, Chroma, Mel)
-Predict emotions such as: Happy, Sad, Neutral
-Emotion-wise probability distribution
-Clean and beginner-friendly UI using Streamlit

## Tech Stack

-Python
-Streamlit
-NumPy
-Pandas
-Scikit-learn
-Librosa
-SoundFile

## Project Structure
.
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies for deployment
├── emotion_recognition.py # Core emotion prediction logic
├── utils.py               # Helper functions
├── features/              # Feature extraction utilities
├── images/                # Screenshots (optional)
└── README.md

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Deployment

This project is deployed using Streamlit Community Cloud and connected directly to this GitHub repository.

## Screenshots

![WhatsApp Image 2026-02-17 at 21 26 26](https://github.com/user-attachments/assets/fd423cb8-e2c2-4162-88d5-3a0debd90256)
![WhatsApp Image 2026-02-17 at 21 27 34](https://github.com/user-attachments/assets/8ea02308-7b26-4fad-a7b1-7d75fe1efdcf)


## Author

Shubhi Singh
B.Tech (CSE) | Machine Learning & Python Enthusiast

## License

MIT License
