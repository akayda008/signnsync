import cv2
import numpy as np
import os
import shutil
from flask import Blueprint, request, jsonify

# Importing preprocessing and model prediction functions
from baara_preprocessing.feature_extract import extract_features
from baara_preprocessing.frame import extract_sharpened_frames
from baara_preprocessing.preprocessing_image import preprocess_images
from baara_preprocessing.sign_prediction import predict_sign_language
from baara_preprocessing.emotion_prediction import predict_emotion

# Flask Blueprint for route handling
routes = Blueprint("routes", __name__)

# ==========================
# 🔹 BASE DIRECTORIES
# ==========================
BASE_PATH = "A:/Softwares/laragon/www/signnsync/interpretation/"
FEATURE_PATH = os.path.join(BASE_PATH, "feature_extracted")
FRAME_PATH = os.path.join(BASE_PATH, "frames")
PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")


# ==========================
# 🔹 FUNCTION TO CLEAR OLD DATA
# ==========================
def clear_old_data():
    """Deletes old extracted files before processing a new video."""
    try:
        for folder in [FEATURE_PATH, FRAME_PATH, PREPROCESSED_PATH]:
            if os.path.exists(folder):
                shutil.rmtree(folder)  # Delete old data
            os.makedirs(folder, exist_ok=True)  # Recreate empty directories

            for subfolder in ["face", "left_hand", "right_hand"]:
                os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
    except Exception as e:
        print(f"⚠️ Error clearing old data: {e}")


# ==========================
# 🔹 VIDEO PROCESSING FUNCTION
# ==========================
def process_video(video_path):
    """
    Extracts features, frames, preprocesses images, and returns extracted file paths.
    This function ensures that preprocessing can call back `routes.py` for final predictions.
    """
    try:
        # Step 1: Feature Extraction (Face + Hands)
        extract_features(video_path, FEATURE_PATH)

        extracted_videos = {
            "face": os.path.join(FEATURE_PATH, "face", "test_face.mp4"),
            "left_hand": os.path.join(FEATURE_PATH, "left_hand", "test_left_hand.mp4"),
            "right_hand": os.path.join(FEATURE_PATH, "right_hand", "test_right_hand.mp4"),
        }

        frame_paths = {}
        for key, vid_path in extracted_videos.items():
            if os.path.exists(vid_path):
                frame_folder = os.path.join(FRAME_PATH, key)
                extract_sharpened_frames(vid_path, frame_folder)
                frame_paths[key] = frame_folder

        preprocessed_paths = {}
        for key, frame_folder in frame_paths.items():
            if os.listdir(frame_folder):
                output_folder = os.path.join(PREPROCESSED_PATH, key)
                preprocess_images(frame_folder, output_folder)
                preprocessed_paths[key] = output_folder

        return preprocessed_paths

    except Exception as e:
        return {"error": str(e)}


# ==========================
# 🔹 EMOTION DETECTION ROUTE
# ==========================
@routes.route("/predict/emotion", methods=["POST"])
def predict_emotion_route():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file received"}), 400

        clear_old_data()

        video_file = request.files["video"]
        test_path = os.path.join(BASE_PATH, "test.mp4")
        video_file.save(test_path)

        if not os.path.exists(test_path):
            return jsonify({"error": "Failed to save uploaded video"}), 500

        preprocessed_paths = process_video(test_path)

        if "face" not in preprocessed_paths:
            return jsonify({"error": "Face detection failed"}), 400

        preprocessed_files = sorted(os.listdir(preprocessed_paths["face"]))
        if not preprocessed_files:
            return jsonify({"error": "Preprocessing failed for face"}), 400

        frame_path = os.path.join(preprocessed_paths["face"], preprocessed_files[0])
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return jsonify({"error": "Invalid preprocessed frame"}), 400

        frame = np.expand_dims(frame, axis=(0, -1))  # Reshape for model
        prediction = predict_emotion(frame)
        predicted_emotion = int(np.argmax(prediction, axis=1)[0])

        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# 🔹 SIGN LANGUAGE RECOGNITION ROUTE
# ==========================
@routes.route("/predict/sign", methods=["POST"])
def predict_sign():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file received"}), 400

        clear_old_data()

        video_file = request.files["video"]
        test_path = os.path.join(BASE_PATH, "test.mp4")
        video_file.save(test_path)

        if not os.path.exists(test_path):
            return jsonify({"error": "Failed to save uploaded video"}), 500

        preprocessed_paths = process_video(test_path)
        response = {}

        for hand in ["left_hand", "right_hand"]:
            if hand in preprocessed_paths:
                preprocessed_files = sorted(os.listdir(preprocessed_paths[hand]))
                if preprocessed_files:
                    frame_path = os.path.join(preprocessed_paths[hand], preprocessed_files[0])
                    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    frame = np.expand_dims(frame, axis=(0, -1))
                    prediction = predict_sign_language(frame)
                    response[hand] = int(np.argmax(prediction, axis=1)[0])
                else:
                    response[hand] = f"Preprocessing failed for {hand}"
            else:
                response[hand] = f"No {hand} detected"

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# 🔹 BOTH (SIGN + EMOTION) ROUTE
# ==========================
@routes.route("/predict/both", methods=["POST"])
def predict_both():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file received"}), 400

        clear_old_data()

        video_file = request.files["video"]
        test_path = os.path.join(BASE_PATH, "test.mp4")
        video_file.save(test_path)

        if not os.path.exists(test_path):
            return jsonify({"error": "Failed to save uploaded video"}), 500

        preprocessed_paths = process_video(test_path)
        response = {}

        # Emotion Prediction
        if "face" in preprocessed_paths:
            preprocessed_files = sorted(os.listdir(preprocessed_paths["face"]))
            if preprocessed_files:
                frame_path = os.path.join(preprocessed_paths["face"], preprocessed_files[0])
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                frame = np.expand_dims(frame, axis=(0, -1))
                prediction = predict_emotion(frame)
                response["emotion"] = int(np.argmax(prediction, axis=1)[0])
            else:
                response["emotion"] = "Preprocessing failed for face"
        else:
            response["emotion"] = "No face detected"

        # Sign Language Prediction
        for hand in ["left_hand", "right_hand"]:
            if hand in preprocessed_paths:
                preprocessed_files = sorted(os.listdir(preprocessed_paths[hand]))
                if preprocessed_files:
                    frame_path = os.path.join(preprocessed_paths[hand], preprocessed_files[0])
                    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    frame = np.expand_dims(frame, axis=(0, -1))
                    prediction = predict_sign_language(frame)
                    response[hand] = int(np.argmax(prediction, axis=1)[0])
                else:
                    response[hand] = "Preprocessing failed"
            else:
                response[hand] = "No hand detected"

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
