import os
import numpy as np
import tensorflow as tf
import cv2
from collections import Counter
from flask import jsonify

# Load trained model
MODEL_PATH = r"C:\Users\avoon\Downloads\model\emotion_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found! Train and save the model first.")

model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels
EMOTIONS = ["Angry", "Anticipation", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised", "Trust"]

# Function to preprocess an image
def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, target_size)  # Resize to match training size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize
    return img

# Function to predict emotion
def predict_emotion(preprocessed_folder):
    frame_preds = []

    for img_name in sorted(os.listdir(preprocessed_folder)):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            img_path = os.path.join(preprocessed_folder, img_name)
            img_array = preprocess_image(img_path)
            pred = model.predict(img_array)
            frame_preds.append(np.argmax(pred))

    if frame_preds:
        most_common_pred = Counter(frame_preds).most_common(1)[0][0]
        predicted_emotion = EMOTIONS[most_common_pred]
    else:
        predicted_emotion = "No valid frames found!"

    return jsonify({"emotion": predicted_emotion})
