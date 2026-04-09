import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_frames

MODEL_PATH = os.path.join("models", "action_model.h5")
LABEL_PATH = os.path.join("models", "class_names.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please run train.py first.")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError("class_names.json not found. Please run train.py first.")

model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


def predict_action(video_path):
    frames = extract_frames(video_path)

    if frames.shape[0] == 0:
        return "Could not read video", 0.0

    frames = np.expand_dims(frames, axis=0)  # (1, MAX_FRAMES, H, W, C)
    prediction = model.predict(frames, verbose=0)[0]

    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    return CLASS_NAMES[class_index], confidence


if __name__ == "__main__":
    sample_video = "sample.avi"
    label, confidence = predict_action(sample_video)
    print("Predicted action:", label)
    print("Confidence:", round(confidence * 100, 2), "%")