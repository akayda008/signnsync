import os
import cv2
import mediapipe as mp
import subprocess  # For calling frame.py
import sys
import time
import numpy as np  # For creating blank frames

# Ensure frame.py is in the same directory or adjust the path
FRAME_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "frame.py")

def extract_features(input_video_path, output_folder):
    """Extracts face, left hand, and right hand from video and saves as separate videos, maintaining original duration."""

    if not os.path.exists(input_video_path):
        print(f"❌ Error: Video file {input_video_path} not found.")
        return

    # Create output folders
    face_video_path = os.path.join(output_folder, "face", "test_face.mp4")
    right_hand_video_path = os.path.join(output_folder, "right_hand", "test_right_hand.mp4")
    left_hand_video_path = os.path.join(output_folder, "left_hand", "test_left_hand.mp4")

    os.makedirs(os.path.dirname(face_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(right_hand_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(left_hand_video_path), exist_ok=True)

    print(f"✅ Created feature extraction folders at: {output_folder}")

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Error: Unable to open video file {input_video_path}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Preserve original FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Processing video: {input_video_path}, FPS: {fps}, Frames: {total_frames}, Resolution: {frame_width}x{frame_height}")

    # Define video writers with correct FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    face_output = cv2.VideoWriter(face_video_path, fourcc, fps, (frame_width, frame_height))
    right_hand_output = cv2.VideoWriter(right_hand_video_path, fourcc, fps, (frame_width, frame_height))
    left_hand_output = cv2.VideoWriter(left_hand_video_path, fourcc, fps, (frame_width, frame_height))

    detected_features = {"face": 0, "right_hand": 0, "left_hand": 0}

    # Create blank frame (black) to maintain frame count
    blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Store last valid frames to prevent flickering
    last_face_frame = blank_frame.copy()
    last_right_hand_frame = blank_frame.copy()
    last_left_hand_frame = blank_frame.copy()

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends
            frame_count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            h, w, _ = frame.shape
            padding = 30  # Padding around detected features

            # Default frames as last detected features
            face_frame = last_face_frame.copy()
            right_hand_frame = last_right_hand_frame.copy()
            left_hand_frame = last_left_hand_frame.copy()

            # Face Extraction
            if results.face_landmarks:
                detected_features["face"] += 1
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in results.face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)
                x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size > 0:
                    face_frame = cv2.resize(face_crop, (frame_width, frame_height))
                    last_face_frame = face_frame.copy()

            # Right Hand Extraction
            if results.right_hand_landmarks:
                detected_features["right_hand"] += 1
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in results.right_hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)
                x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                right_hand_crop = frame[y_min:y_max, x_min:x_max]
                if right_hand_crop.size > 0:
                    right_hand_frame = cv2.resize(right_hand_crop, (frame_width, frame_height))
                    last_right_hand_frame = right_hand_frame.copy()

            # Left Hand Extraction
            if results.left_hand_landmarks:
                detected_features["left_hand"] += 1
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in results.left_hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)
                x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                left_hand_crop = frame[y_min:y_max, x_min:x_max]
                if left_hand_crop.size > 0:
                    left_hand_frame = cv2.resize(left_hand_crop, (frame_width, frame_height))
                    last_left_hand_frame = left_hand_frame.copy()

            # Write frames to maintain original FPS and duration
            face_output.write(face_frame)
            right_hand_output.write(right_hand_frame)
            left_hand_output.write(left_hand_frame)

            # ⏳ Maintain original frame duration
            elapsed_time = time.time() - start_time
            expected_time = frame_count / fps
            if elapsed_time < expected_time:
                time.sleep(expected_time - elapsed_time)

    # Release everything
    cap.release()
    face_output.release()
    right_hand_output.release()
    left_hand_output.release()

    print(f"✅ Feature extraction complete: Face ({detected_features['face']} frames), Right Hand ({detected_features['right_hand']} frames), Left Hand ({detected_features['left_hand']} frames).")

    # Call frame.py to process extracted videos
    print("🚀 Calling frame.py for frame extraction...")
    subprocess.run([sys.executable, FRAME_SCRIPT_PATH, face_video_path, right_hand_video_path, left_hand_video_path])

# Example usage
if __name__ == "__main__":
    extract_features("input_video.mp4", "output_folder")
