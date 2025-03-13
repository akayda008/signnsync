import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from collections import Counter
from flask import jsonify

# Load trained model
MODEL_PATH = r"A:/Softwares/laragon/www/signnsync/flask_api/model/sign_language_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Sign language model file not found!")

model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (ensure these are correct)
CLASS_LABELS = ["Hello", "Thank You", "Yes", "No", "I Love You", "Help", "Sorry", "Please", "Stop"]

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
        img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"⚠️ Error processing image {img_path}: {str(e)}")
        return None

# Function to predict sign language
def predict_sign_language(left_hand_folder, right_hand_folder):
    """
    Predicts the sign language gesture based on left-hand and right-hand images.

    :param left_hand_folder: Path to left-hand images
    :param right_hand_folder: Path to right-hand images
    :return: JSON response with predicted sign language gesture
    """

    left_hand_preds = []
    right_hand_preds = []

    # Validate folders
    if not os.path.exists(left_hand_folder) or not os.listdir(left_hand_folder):
        return jsonify({"error": "⚠️ No left-hand images found!"})
    
    if not os.path.exists(right_hand_folder) or not os.listdir(right_hand_folder):
        return jsonify({"error": "⚠️ No right-hand images found!"})

    # Process left-hand images
    for img_name in sorted(os.listdir(left_hand_folder)):
        if img_name.endswith((".png", ".jpg", ".jpeg")):  # Process only valid images
            img_path = os.path.normpath(os.path.join(left_hand_folder, img_name))
            img_array = preprocess_image(img_path)
            
            if img_array is not None:
                left_pred = model.predict(img_array)
                left_hand_preds.append(np.argmax(left_pred[0]))

    # Process right-hand images
    for img_name in sorted(os.listdir(right_hand_folder)):
        if img_name.endswith((".png", ".jpg", ".jpeg")):  # Process only valid images
            img_path = os.path.normpath(os.path.join(right_hand_folder, img_name))
            img_array = preprocess_image(img_path)
            
            if img_array is not None:
                right_pred = model.predict(img_array)
                right_hand_preds.append(np.argmax(right_pred[0]))

    # Get most common predictions
    final_left_pred = Counter(left_hand_preds).most_common(1)[0][0] if left_hand_preds else None
    final_right_pred = Counter(right_hand_preds).most_common(1)[0][0] if right_hand_preds else None

    # Ensure predictions match
    if final_left_pred is not None and final_right_pred is not None and final_left_pred == final_right_pred:
        prediction = CLASS_LABELS[final_right_pred]
    else:
        prediction = "This sign is not available in the database."

    return jsonify({"sign_language": prediction})
