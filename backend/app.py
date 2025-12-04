from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import uvicorn

# Load YAMNet from TensorFlow Hub
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# Load your trained 3-class classifier
model = tf.keras.models.load_model("babycry_3class_yamnet_model.keras")

# Class names
LABELS = ["hunger", "pain", "discomfort"]

app = FastAPI()

# Allow frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_embedding(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    scores, embeddings, spectrogram = yamnet(audio)
    return np.mean(embeddings.numpy(), axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Extract embedding
    embedding = extract_embedding(audio_path)
    embedding = np.expand_dims(embedding, axis=0)

    # Predict
    preds = model.predict(embedding)[0]
    index = np.argmax(preds)
    confidence = float(preds[index])
    label = LABELS[index]

    return {
        "prediction": label,
        "confidence": confidence
    }

@app.get("/")
def home():
    return {"message": "Baby Cry Classifier API is running!"}

# Only for local run
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
