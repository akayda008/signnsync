import cv2
import numpy as np
import os
from flask import Blueprint, request, jsonify
from preprocessing.feature_extract import extract_features_from_frame
from preprocessing.extract_frames import extract_frames
from preprocessing.preprocess_image import preprocess_image
from model_loader import emotion_model, sign_model

routes = Blueprint("routes", __name__)

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
        
        # Save video temporarily
        temp_video_path = "A:/Softwares/laragon/www/signnsync/video/temp_video.mp4"
        video_file.save(temp_video_path)

        print("✅ Video received and saved successfully.")

        # Extract frames from video
        frames = extract_frames(temp_video_path)

        if not frames:
            os.remove(temp_video_path)
            return jsonify({"error": "No frames extracted from video"}), 400

        print(f"📸 Extracted {len(frames)} frames.")

        # Process the first frame for emotion detection
        frame = frames[0]
        face, _, _ = extract_features_from_frame(frame)

        if face is None:
            os.remove(temp_video_path)
            return jsonify({"error": "No face detected"}), 400

        face = preprocess_image(face)

        # Ensure correct input shape
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict
        prediction = emotion_model.predict(face)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        os.remove(temp_video_path)
        return jsonify({"emotion": predicted_class})

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
        
        # Save video temporarily
        temp_video_path = "A:/Softwares/laragon/www/signnsync/video/temp_video.mp4"
        video_file.save(temp_video_path)

        print("✅ Video received and saved successfully.")

        # Extract frames from video
        frames = extract_frames(temp_video_path)

        if not frames:
            os.remove(temp_video_path)
            return jsonify({"error": "No frames extracted from video"}), 400

        print(f"📸 Extracted {len(frames)} frames.")

        # Process the first frame for sign language recognition
        frame = frames[0]
        _, left_hand, right_hand = extract_features_from_frame(frame)

        response = {}

        # If left hand is detected, process it
        if left_hand is not None:
            left_hand = preprocess_image(left_hand)
            left_hand = np.expand_dims(left_hand, axis=0)  # Add batch dimension
            left_prediction = sign_model.predict(left_hand)
            response["left_hand"] = int(np.argmax(left_prediction, axis=1)[0])
        else:
            response["left_hand"] = "No left hand detected"

        # If right hand is detected, process it
        if right_hand is not None:
            right_hand = preprocess_image(right_hand)
            right_hand = np.expand_dims(right_hand, axis=0)  # Add batch dimension
            right_prediction = sign_model.predict(right_hand)
            response["right_hand"] = int(np.argmax(right_prediction, axis=1)[0])
        else:
            response["right_hand"] = "No right hand detected"

        os.remove(temp_video_path)
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
        
        # Save video temporarily
        temp_video_path = "A:/Softwares/laragon/www/signnsync/video/temp_video.mp4"
        video_file.save(temp_video_path)

        print("✅ Video received and saved successfully.")

        # Extract frames from video
        frames = extract_frames(temp_video_path)

        if not frames:
            os.remove(temp_video_path)
            return jsonify({"error": "No frames extracted from video"}), 400

        print(f"📸 Extracted {len(frames)} frames.")

        # Process the first frame for both tasks
        frame = frames[0]
        face, left_hand, right_hand = extract_features_from_frame(frame)

        response = {}

        # Emotion Detection
        if face is not None:
            face = preprocess_image(face)
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            emotion_prediction = emotion_model.predict(face)
            response["emotion"] = int(np.argmax(emotion_prediction, axis=1)[0])
        else:
            response["emotion"] = "No face detected"

        # Sign Language Recognition (process left hand, right hand, or both)
        if left_hand is not None:
            left_hand = preprocess_image(left_hand)
            left_hand = np.expand_dims(left_hand, axis=0)  # Add batch dimension
            left_prediction = sign_model.predict(left_hand)
            response["left_hand"] = int(np.argmax(left_prediction, axis=1)[0])
        else:
            response["left_hand"] = "No left hand detected"

        if right_hand is not None:
            right_hand = preprocess_image(right_hand)
            right_hand = np.expand_dims(right_hand, axis=0)  # Add batch dimension
            right_prediction = sign_model.predict(right_hand)
            response["right_hand"] = int(np.argmax(right_prediction, axis=1)[0])
        else:
            response["right_hand"] = "No right hand detected"

        os.remove(temp_video_path)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
