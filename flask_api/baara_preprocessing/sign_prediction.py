import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import jsonify

# Load trained model
MODEL_PATH = r"C:/Users/Baarathi J V/Downloads/6_sem-Project/sign_language_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found! Train and save the model first.")

model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = ["Angry", "Disgust", "Happy", "Trust", "Surprised", "Fear", "Sad", "Hope", "Neutral"]

# Function to preprocess an image
def preprocess_image(img_path, target_size=(64, 64)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Function to predict sign language
def predict_sign_language(left_hand_folder, right_hand_folder):
    left_hand_preds = []
    right_hand_preds = []

    for img_name in sorted(os.listdir(left_hand_folder)):
        img_path = os.path.join(left_hand_folder, img_name)
        img_array = preprocess_image(img_path)
        left_pred = model.predict(img_array)
        left_hand_preds.append(np.argmax(left_pred))

    for img_name in sorted(os.listdir(right_hand_folder)):
        img_path = os.path.join(right_hand_folder, img_name)
        img_array = preprocess_image(img_path)
        right_pred = model.predict(img_array)
        right_hand_preds.append(np.argmax(right_pred))

    final_left_pred = max(set(left_hand_preds), key=left_hand_preds.count)
    final_right_pred = max(set(right_hand_preds), key=right_hand_preds.count)

    if final_left_pred == final_right_pred:
        prediction = CLASS_LABELS[final_right_pred]
    else:
        prediction = "This sign is not available in the database."

    return jsonify({"sign_language": prediction})
