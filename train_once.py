from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
import joblib

model = SVC(probability=True)

rec = EmotionRecognizer(
    model=model,
    emotions=["sad", "neutral", "happy"],
    balance=True,
    verbose=1
)

rec.train()

joblib.dump(rec, "emotion_model.pkl")
print("Model saved as emotion_model.pkl")
