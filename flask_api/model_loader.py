import os
import tensorflow as tf

# Get the absolute path to the 'model' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")

# Paths to model files
emotion_model_path = os.path.join(MODEL_PATH, "emotion_model.h5")
sign_model_path = os.path.join(MODEL_PATH, "sign_language_model.h5")

# Check if files exist before loading
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"Emotion model not found at {emotion_model_path}")

if not os.path.exists(sign_model_path):
    raise FileNotFoundError(f"Sign model not found at {sign_model_path}")

# Load the models
emotion_model = tf.keras.models.load_model(emotion_model_path)
sign_model = tf.keras.models.load_model(sign_model_path)
