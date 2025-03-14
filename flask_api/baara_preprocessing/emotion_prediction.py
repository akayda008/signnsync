import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image

# ==========================
# 🔹 Suppress TensorFlow Warnings (Optional)
# ==========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress unnecessary TF logs

# ==========================
# 🔹 Load Trained Model
# ==========================
MODEL_PATH = r"A:/Softwares/laragon/www/signnsync/flask_api/model/emotion_model.h5"

if not os.path.exists(MODEL_PATH):
    print(json.dumps({"error": "❌ Emotion model file not found! Train and save the model first."}))
    exit()

# ✅ Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# 🔹 Function to Preprocess Image
# ==========================
def preprocess_image(img_path, target_size=(64, 64)):
    """Loads and preprocesses an image for emotion detection."""
    try:
        img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        return img_array
    except Exception as e:
        return {"error": f"❌ Image preprocessing failed: {str(e)}"}

# ==========================
# 🔹 Function to Predict Emotion
# ==========================
def predict_emotion():
    input_folder = r"A:/Softwares/laragon/www/signnsync/interpretation/preprocessed"
    face_folder = os.path.join(input_folder, "face")

    emotion_preds = []

    # ✅ Ensure input folder exists and contains frames
    if not os.path.exists(face_folder) or not os.listdir(face_folder):
        return json.dumps({"error": "❌ No preprocessed face frames found!"})

    try:
        # ✅ Predict for face frames
        for img_name in sorted(os.listdir(face_folder)):  # Ensure ordered processing
            img_path = os.path.join(face_folder, img_name)
            img_array = preprocess_image(img_path)
            
            if isinstance(img_array, dict) and "error" in img_array:
                return json.dumps(img_array)  # Return JSON error if preprocessing fails

            # ✅ Suppress TensorFlow verbose logs
            tf.get_logger().setLevel("ERROR")

            # ✅ Make prediction
            emotion_pred = model.predict(img_array, verbose=0)  # Suppressed verbose output
            emotion_preds.append(np.argmax(emotion_pred))

        # ✅ Ensure at least one prediction is made
        if not emotion_preds:
            return json.dumps({"error": "❌ No valid predictions made!"})

        # ✅ Get the most frequent prediction
        final_emotion_pred = max(set(emotion_preds), key=emotion_preds.count)

        # ✅ Class labels (Modify this if needed)
        class_labels = ["Angry", "Anticipation", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised", "Trust"]

        # ✅ Determine final prediction text
        prediction_text = class_labels[final_emotion_pred]

        return json.dumps({"emotion_prediction": prediction_text})

    except Exception as e:
        return json.dumps({"error": f"⚠️ Prediction error: {str(e)}"})

# ==========================
# 🔹 Run Prediction
# ==========================
if __name__ == "__main__":
    print(predict_emotion())