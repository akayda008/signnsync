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
MODEL_PATH = r"A:/Softwares/laragon/www/signnsync/flask_api/model/sign_language_model.h5"

if not os.path.exists(MODEL_PATH):
    print(json.dumps({"error": "❌ Model file not found! Train and save the model first."}))
    exit()

# ✅ Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# 🔹 Function to Preprocess Image
# ==========================
def preprocess_image(img_path, target_size=(64, 64)):
    """Loads and preprocesses an image for model prediction."""
    try:
        img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        return img_array
    except Exception as e:
        return {"error": f"❌ Image preprocessing failed: {str(e)}"}

# ==========================
# 🔹 Function to Predict Sign Language
# ==========================
def predict_sign_language():
    input_folder = r"A:/Softwares/laragon/www/signnsync/interpretation/preprocessed"
    left_hand_folder = os.path.join(input_folder, "left_hand")
    right_hand_folder = os.path.join(input_folder, "right_hand")

    left_hand_preds = []
    right_hand_preds = []

    # ✅ Ensure both input folders exist and contain frames
    if not os.path.exists(left_hand_folder) or not os.listdir(left_hand_folder):
        return json.dumps({"error": "❌ No preprocessed left-hand frames found!"})

    if not os.path.exists(right_hand_folder) or not os.listdir(right_hand_folder):
        return json.dumps({"error": "❌ No preprocessed right-hand frames found!"})

    try:
        # ✅ Predict for left hand
        for img_name in sorted(os.listdir(left_hand_folder)):
            img_path = os.path.join(left_hand_folder, img_name)
            img_array = preprocess_image(img_path)

            if isinstance(img_array, dict) and "error" in img_array:
                return json.dumps(img_array)  # Return JSON error if preprocessing fails

            # ✅ Suppress TensorFlow verbose logs
            tf.get_logger().setLevel("ERROR")

            # ✅ Make prediction
            left_pred = model.predict(img_array, verbose=0)  # Suppressed verbose output
            left_hand_preds.append(np.argmax(left_pred))

        # ✅ Predict for right hand
        for img_name in sorted(os.listdir(right_hand_folder)):
            img_path = os.path.join(right_hand_folder, img_name)
            img_array = preprocess_image(img_path)

            if isinstance(img_array, dict) and "error" in img_array:
                return json.dumps(img_array)  # Return JSON error if preprocessing fails

            # ✅ Make prediction
            right_pred = model.predict(img_array, verbose=0)
            right_hand_preds.append(np.argmax(right_pred))

        # ✅ Ensure at least one prediction is made
        if not left_hand_preds and not right_hand_preds:
            return json.dumps({"error": "❌ No valid predictions made!"})

        # ✅ Determine the most frequent prediction
        final_left_pred = max(set(left_hand_preds), key=left_hand_preds.count) if left_hand_preds else None
        final_right_pred = max(set(right_hand_preds), key=right_hand_preds.count) if right_hand_preds else None

        # ✅ Class labels (Modify this as needed)
        class_labels = ["Angry", "Disgust", "Happy", "Trust", "Surprised", "Fear", "Sad", "Hope", "Neutral"]

        # ✅ Determine final prediction text
        if final_left_pred is not None and final_right_pred is not None:
            if final_left_pred == final_right_pred:
                prediction_text = class_labels[final_right_pred]
            else:
                prediction_text = "⚠️ Left and right hands detected different signs."
        elif final_left_pred is not None:
            prediction_text = class_labels[final_left_pred]
        elif final_right_pred is not None:
            prediction_text = class_labels[final_right_pred]
        else:
            prediction_text = "❌ No valid sign detected."

        return json.dumps({"sign_prediction": prediction_text})

    except Exception as e:
        return json.dumps({"error": f"⚠️ Prediction error: {str(e)}"})

# ==========================
# 🔹 Run Prediction
# ==========================
if __name__ == "__main__":
    print(predict_sign_language())