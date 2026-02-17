from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC

# create a simple SVM model
model = SVC(probability=True)

# initialize emotion recognizer
rec = EmotionRecognizer(
    model=model,
    emotions=["sad", "neutral", "happy"],
    balance=True,
    verbose=1
)

# train the model
rec.train()

# print accuracy
print("Train accuracy:", rec.train_score())
print("Test accuracy:", rec.test_score())
# ----------- AUDIO PREDICTION -----------
audio_path = "data/validation/Actor_07/03-02-06-01-01-01-07_fear.wav"


prediction = rec.predict(audio_path)
print("Predicted emotion:", prediction)