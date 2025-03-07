import os
import numpy as np
import cv2
from flask import Blueprint, request, jsonify
from baara_preprocessing.feature_extract import extract_features # ✅ Corrected Import
from baara_preprocessing.frame import run_frame_extraction
from baara_preprocessing.preprocessing_image import preprocess_images
from model_loader import emotion_model, sign_model

routes = Blueprint("routes", __name__)

TEMP_VIDEO_PATH = "A:/Softwares/laragon/www/signnsync/video/temp_video.mp4"

# ===========================
# 🔹 EMOTION DETECTION ROUTE (Updated)
# ===========================
@routes.route("/predict/emotion", methods=["POST"])
def predict_emotion():
    try:
        if "video" not in request.files:
            print("❌ No video file received!")
            return jsonify({"error": "No video file received"}), 400

        video_file = request.files["video"]
        video_file.save(TEMP_VIDEO_PATH)
        print("✅ Video received and saved successfully.")

        # ✅ Extract features using the new function
        features = extract_features(TEMP_VIDEO_PATH)
        face = features.get("face")

        if face is None:
            os.remove(TEMP_VIDEO_PATH)
            return jsonify({"error": "No face detected in video"}), 400

        # ✅ Preprocess the face
        face = preprocess_images(face)
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # ✅ Predict emotion
        prediction = emotion_model.predict(face)
        final_prediction = np.argmax(prediction, axis=1)[0]

        os.remove(TEMP_VIDEO_PATH)
        return jsonify({"emotion": final_prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================
# 🔹 SIGN LANGUAGE RECOGNITION ROUTE (Updated)
# ===========================
@routes.route("/predict/sign", methods=["POST"])
def predict_sign():
    try:
        if "video" not in request.files:
            print("❌ No video file received!")
            return jsonify({"error": "No video file received"}), 400

        video_file = request.files["video"]
        video_file.save(TEMP_VIDEO_PATH)
        print("✅ Video received and saved successfully.")

        # ✅ Extract features
        features = extract_features(TEMP_VIDEO_PATH)
        left_hand = features.get("left_hand")
        right_hand = features.get("right_hand")

        left_predictions, right_predictions = [], []

        # ✅ Process left hand
        if left_hand is not None:
            left_hand = preprocess_images(left_hand)
            left_hand = np.expand_dims(left_hand, axis=0)
            left_predictions.append(np.argmax(sign_model.predict(left_hand), axis=1)[0])

        # ✅ Process right hand
        if right_hand is not None:
            right_hand = preprocess_images(right_hand)
            right_hand = np.expand_dims(right_hand, axis=0)
            right_predictions.append(np.argmax(sign_model.predict(right_hand), axis=1)[0])

        os.remove(TEMP_VIDEO_PATH)

        response = {
            "left_hand": max(set(left_predictions), key=left_predictions.count) if left_predictions else "No left hand detected",
            "right_hand": max(set(right_predictions), key=right_predictions.count) if right_predictions else "No right hand detected"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================
# 🔹 BOTH (SIGN + EMOTION) ROUTE (Updated)
# ===========================
@routes.route("/predict/both", methods=["POST"])
def predict_both():
    try:
        if "video" not in request.files:
            print("❌ No video file received!")
            return jsonify({"error": "No video file received"}), 400

        video_file = request.files["video"]
        video_file.save(TEMP_VIDEO_PATH)
        print("✅ Video received and saved successfully.")

        # ✅ Extract features
        features = extract_features(TEMP_VIDEO_PATH)
        face = features.get("face")
        left_hand = features.get("left_hand")
        right_hand = features.get("right_hand")

        emotion_predictions, left_predictions, right_predictions = [], [], []

        # ✅ Process face for emotion detection
        if face is not None:
            face = preprocess_images(face)
            face = np.expand_dims(face, axis=0)
            emotion_predictions.append(np.argmax(emotion_model.predict(face), axis=1)[0])

        # ✅ Process left hand for sign language
        if left_hand is not None:
            left_hand = preprocess_images(left_hand)
            left_hand = np.expand_dims(left_hand, axis=0)
            left_predictions.append(np.argmax(sign_model.predict(left_hand), axis=1)[0])

        # ✅ Process right hand for sign language
        if right_hand is not None:
            right_hand = preprocess_images(right_hand)
            right_hand = np.expand_dims(right_hand, axis=0)
            right_predictions.append(np.argmax(sign_model.predict(right_hand), axis=1)[0])

        os.remove(TEMP_VIDEO_PATH)

        response = {
            "emotion": max(set(emotion_predictions), key=emotion_predictions.count) if emotion_predictions else "No face detected",
            "left_hand": max(set(left_predictions), key=left_predictions.count) if left_predictions else "No left hand detected",
            "right_hand": max(set(right_predictions), key=right_predictions.count) if right_predictions else "No right hand detected"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================
# 🔹 ADDITIONAL FEATURE EXTRACTION ROUTES
# ===========================

feature_bp = Blueprint("feature", __name__)

@feature_bp.route("/extract_features", methods=["POST"])
def feature_extraction():
    data = request.get_json()
    input_folder = data.get("input_folder")
    output_folder = data.get("output_folder")

    if not input_folder or not output_folder:
        return jsonify({"error": "Both input_folder and output_folder are required"}), 400

    try:
        extract_features(input_folder, output_folder)
        return jsonify({"message": "Feature extraction completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


video_processing = Blueprint("video_processing", __name__)

@video_processing.route("/extract_frames", methods=["POST"])
def run_frame_extraction_api():
    base_input = "A:/Christ/Academics/CIA/CS Project/Data/ISL/Trust/Feature_Extract_trust"
    base_output = "A:/Christ/Academics/CIA/CS Project/Data/ISL/Trust/Frames_trust"
    run_frame_extraction(base_input, base_output)
    return jsonify({"message": "Frame extraction completed!"})

@video_processing.route("/preprocess_images", methods=["POST"])
def preprocess_images_api():
    try:
        preprocess_images()
        return jsonify({"message": "Image preprocessing completed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
