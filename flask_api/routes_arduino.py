import os
import shutil
import subprocess
import json
import serial
import time
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
VIDEO_PATH = os.path.join(BASE_PATH, "test.mp4")

# ==========================
# 🔹 ARDUINO SERIAL CONFIGURATION
# ==========================
SERIAL_PORT = "COM3"  # Change this to match your Arduino port
BAUD_RATE = 9600

def send_to_arduino(command):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            ser.write(f"{command}\n".encode())
            time.sleep(1)
    except Exception as e:
        print(f"⚠️ Error communicating with Arduino: {e}")

# ==========================
# 🔹 FUNCTION TO CLEAR OLD DATA
# ==========================
def clear_old_data():
    """Deletes old extracted files before processing a new video."""
    try:
        for folder in [FEATURE_PATH, FRAME_PATH, PREPROCESSED_PATH]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            for subfolder in ["face", "left_hand", "right_hand"]:
                os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
        print("✅ Old data cleared successfully.")
    except Exception as e:
        print(f"⚠️ Error clearing old data: {e}")

# ==========================
# 🔹 VIDEO PROCESSING FUNCTION
# ==========================
def process_video():
    """Extracts features, frames, preprocesses images, and returns extracted file paths."""
    try:
        extract_features(VIDEO_PATH, FEATURE_PATH)
       
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

        print("✅ Video processing completed successfully.")
        return preprocessed_paths
    except Exception as e:
        print(f"⚠️ Error processing video: {e}")
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
        result = subprocess.run([
            "python", script_file
        ], cwd=SCRIPT_PATH, capture_output=True, text=True)

        if result.returncode != 0:
            return {"error": f"⚠️ Error in {script_name}: {result.stderr.strip()}"}

        return json.loads(result.stdout.strip())
    except Exception as e:
        return {"error": f"⚠️ Exception running {script_name}: {str(e)}"}

# ==========================
# 🔹 ARDUINO START RECORDING ROUTE
# ==========================
@routes.route("/arduino/start", methods=["POST"])
def arduino_start():
    try:
        clear_old_data()
        send_to_arduino("START")
        return jsonify({"message": "✅ Recording started."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================
# 🔹 ARDUINO STOP RECORDING & PROCESS ROUTE
# ==========================
@routes.route("/arduino/stop", methods=["POST"])
def arduino_stop():
    try:
        send_to_arduino("STOP")
       
        if not os.path.exists(VIDEO_PATH):
            return jsonify({"error": "❌ No recorded video found."}), 500
       
        process_video()
       
        emotion_result = run_prediction_script("emotion_prediction.py")
        sign_result = run_prediction_script("sign_prediction.py")
       
        return jsonify({
            "emotion": emotion_result.get("emotion", "No face detected"),
            "left_hand": sign_result.get("left_hand", "No left hand detected"),
            "right_hand": sign_result.get("right_hand", "No right hand detected")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500