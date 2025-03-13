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
    print(json.dumps({"error": "❌ Model file not found! Train and save the model first."}, indent=4))
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
        return None, f"❌ Image preprocessing failed: {str(e)}"

# ==========================
# 🔹 Function to Predict Sign Language
# ==========================
def predict_sign_language():
    input_folder = r"A:/Softwares/laragon/www/signnsync/interpretation/preprocessed"
    left_hand_folder = os.path.join(input_folder, "left_hand")
    right_hand_folder = os.path.join(input_folder, "right_hand")

    left_hand_preds = []
    right_hand_preds = []

    # ✅ Ensure both input folders exist
    if not os.path.exists(left_hand_folder) or not os.listdir(left_hand_folder):
        return json.dumps({"error": "❌ No preprocessed left-hand frames found!"}, indent=4)

    if not os.path.exists(right_hand_folder) or not os.listdir(right_hand_folder):
        return json.dumps({"error": "❌ No preprocessed right-hand frames found!"}, indent=4)

    try:
        # ✅ Predict for left hand
        for img_name in sorted(os.listdir(left_hand_folder)):
            img_path = os.path.join(left_hand_folder, img_name)
            img_array, error_msg = preprocess_image(img_path)

            if img_array is None:
                return json.dumps({"error": error_msg}, indent=4)  # Return JSON error if preprocessing fails

            # ✅ Make prediction
            pred = model.predict(img_array, verbose=0)
            left_hand_preds.append(np.argmax(pred))

        # ✅ Predict for right hand
        for img_name in sorted(os.listdir(right_hand_folder)):
            img_path = os.path.join(right_hand_folder, img_name)
            img_array, error_msg = preprocess_image(img_path)

            if img_array is None:
                return json.dumps({"error": error_msg}, indent=4)  # Return JSON error if preprocessing fails

            # ✅ Make prediction
            pred = model.predict(img_array, verbose=0)
            right_hand_preds.append(np.argmax(pred))

        # ✅ Ensure at least one prediction is made
        if not left_hand_preds and not right_hand_preds:
            return json.dumps({"error": "❌ No valid predictions made!"}, indent=4)

        # ✅ Determine the most frequent prediction
        final_left_pred = max(set(left_hand_preds), key=left_hand_preds.count) if left_hand_preds else None
        final_right_pred = max(set(right_hand_preds), key=right_hand_preds.count) if right_hand_preds else None

        # ✅ Class labels (Modify this as needed)
        class_labels = ["Hello", "Thank You", "Yes", "No", "Please", "Sorry", "Goodbye", "I Love You", "Help"]

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

        return json.dumps({"sign_prediction": prediction_text}, indent=4)

    except Exception as e:
        return json.dumps({"error": f"⚠️ Prediction error: {str(e)}"}, indent=4)

# ==========================
# 🔹 Run Prediction
# ==========================
if __name__ == "__main__":
    print(predict_sign_language())
