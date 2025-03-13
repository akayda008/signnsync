import os
import numpy as np
import tensorflow as tf
import cv2
from collections import Counter
from flask import jsonify

# Load trained model
MODEL_PATH = r"A:/Softwares/laragon/www/signnsync/flask_api/model/emotion_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Emotion model file not found!")

model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels
EMOTIONS = ["Angry", "Anticipation", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised", "Trust"]

# Function to preprocess an image
def preprocess_image(img_path, target_size=(64, 64)):
    """
    Preprocesses an image:
    - Loads the image in grayscale
    - Resizes to target size (64x64)
    - Normalizes pixel values (0-1)

    :param img_path: Path to image
    :param target_size: Target size for model input (default: (64,64))
    :return: Preprocessed image array
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is None:
            print(f"⚠️ Invalid image: {img_path}")
            return None  # Skip invalid images
        
        img = cv2.resize(img, target_size)  # Resize to match model input size
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = img / 255.0  # Normalize pixel values
        return img
    except Exception as e:
        print(f"⚠️ Error processing image {img_path}: {str(e)}")
        return None

# Function to predict emotion
def predict_emotion(preprocessed_folder):
    """
    Predicts the dominant emotion based on processed face images.

    :param preprocessed_folder: Path to folder containing preprocessed face images
    :return: JSON response with predicted emotion
    """

    frame_preds = []

    # Validate folder
    if not os.path.exists(preprocessed_folder) or not os.listdir(preprocessed_folder):
        return jsonify({"error": "⚠️ No face images found!"})

    for img_name in sorted(os.listdir(preprocessed_folder)):
        if img_name.endswith((".png", ".jpg", ".jpeg")):  # Process only valid images
            img_path = os.path.normpath(os.path.join(preprocessed_folder, img_name))
            img_array = preprocess_image(img_path)
            
            if img_array is not None:
                pred = model.predict(img_array)
                frame_preds.append(np.argmax(pred[0]))

    # Get the most common prediction
    if frame_preds:
        most_common_pred = Counter(frame_preds).most_common(1)[0][0]
        predicted_emotion = EMOTIONS[most_common_pred]
    else:
        predicted_emotion = "No valid frames found!"

    return jsonify({"emotion": predicted_emotion})
