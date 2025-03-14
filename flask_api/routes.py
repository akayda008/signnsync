import os
import shutil
import subprocess
import json
from flask import Blueprint, request, jsonify

# Importing preprocessing functions
from baara_preprocessing.feature_extract import extract_features
from baara_preprocessing.frame import extract_sharpened_frames
from baara_preprocessing.preprocessing_image import preprocess_images

# Flask Blueprint for routes
routes = Blueprint("routes", __name__)

# ==========================
# 🔹 BASE DIRECTORIES
# ==========================
BASE_PATH = "A:/Softwares/laragon/www/signnsync/interpretation/"
FEATURE_PATH = os.path.join(BASE_PATH, "feature_extracted")
FRAME_PATH = os.path.join(BASE_PATH, "frames")
PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
SCRIPT_PATH = "A:/Softwares/laragon/www/signnsync/flask_api/baara_preprocessing"

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
    """Extracts features, frames, preprocesses images, and returns extracted file paths."""
    try:
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
# 🔹 FUNCTION TO RUN PREDICTION SCRIPTS
# ==========================
def run_prediction_script(script_name):
    """Runs a prediction script and returns its JSON output."""
    script_file = os.path.join(SCRIPT_PATH, script_name)
    
    if not os.path.exists(script_file):
        return {"error": f"❌ Script {script_name} not found at {SCRIPT_PATH}"}

    try:
        result = subprocess.run(
            ["python", script_file],
            cwd=SCRIPT_PATH,  
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return {"error": f"⚠️ Error in {script_name}: {result.stderr.strip()}"}

        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {"error": f"⚠️ Invalid JSON output from {script_name}: {result.stdout.strip()}"}

    except Exception as e:
        return {"error": f"⚠️ Exception running {script_name}: {str(e)}"}

# ==========================
# 🔹 EMOTION DETECTION ROUTE
# ==========================
@routes.route("/predict/emotion", methods=["POST"])
def predict_emotion_route():
    try:
        if "video" not in request.files:
            return jsonify({"error": "❌ No video file received"}), 400

        clear_old_data()

        video_file = request.files["video"]
        test_path = os.path.join(BASE_PATH, "test.mp4")
        video_file.save(test_path)

        if not os.path.exists(test_path):
            return jsonify({"error": "❌ Failed to save uploaded video"}), 500

        process_video(test_path)
        emotion_result = run_prediction_script("emotion_prediction.py")

        return jsonify(emotion_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================
# 🔹 SIGN LANGUAGE DETECTION ROUTE
# ==========================
@routes.route("/predict/sign", methods=["POST"])
def predict_sign_route():
    try:
        if "video" not in request.files:
            return jsonify({"error": "❌ No video file received"}), 400

        clear_old_data()

        video_file = request.files["video"]
        test_path = os.path.join(BASE_PATH, "test.mp4")
        video_file.save(test_path)

        if not os.path.exists(test_path):
            return jsonify({"error": "❌ Failed to save uploaded video"}), 500

        process_video(test_path)
        sign_result = run_prediction_script("sign_prediction.py")

        return jsonify(sign_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================
# 🔹 BOTH (SIGN + EMOTION) ROUTE
# ==========================
@routes.route("/predict/both", methods=["POST"])
def predict_both_route():
    try:
        if "video" not in request.files:
            return jsonify({"error": "❌ No video file received"}), 400

        clear_old_data()

        video_file = request.files["video"]
        test_path = os.path.join(BASE_PATH, "test.mp4")
        video_file.save(test_path)

        if not os.path.exists(test_path):
            return jsonify({"error": "❌ Failed to save uploaded video"}), 500

        process_video(test_path)

        # Run both predictions
        emotion_result = run_prediction_script("emotion_prediction.py")
        sign_result = run_prediction_script("sign_prediction.py")

        return jsonify({
            "emotion_prediction_output": emotion_result,
            "sign_prediction_output": sign_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500