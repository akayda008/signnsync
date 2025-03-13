import cv2
import os
import subprocess  # To call preprocessing_image.py
import sys

# Ensure preprocess_image.py is in the same directory or adjust the path
PREPROCESS_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "preprocessing_image.py")

def extract_sharpened_frames(input_folder, output_folder, frame_rate=5):
    """
    Extracts frames from videos in input_folder, applies sharpening, and saves them directly in output_folder.

    Args:
        input_folder (str): Directory containing input video files.
        output_folder (str): Directory where extracted frames will be saved.
        frame_rate (int): Number of frames to extract per second.
    """

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for video_file in os.listdir(input_folder):
        if video_file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.normpath(os.path.join(input_folder, video_file))
            video_name = os.path.splitext(video_file)[0]  # Extract name without extension

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0 or not cap.isOpened():
                print(f"⚠️ Skipping {video_file}: Invalid video file or FPS = 0")
                cap.release()
                continue

            frame_interval = max(1, int(fps / frame_rate))  # Corrected frame interval calculation
            frame_count = 0
            saved_count = 0

            print(f"🎥 Processing video: {video_file} ({fps:.2f} FPS, extracting every {frame_interval} frames)")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Extract 5 frames per second
                if frame_count % frame_interval == 0:
                    # Apply Sharpening to Reduce Motion Blur
                    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                    sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

                    # Save sharpened frame directly in output_folder
                    frame_filename = os.path.normpath(os.path.join(output_folder, f"{video_name}_frame_{saved_count:04d}.jpg"))
                    cv2.imwrite(frame_filename, sharpened)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"✅ {saved_count} frames extracted and saved in: {output_folder}")

# 🔹 Define paths
base_input_folder = r"A:/Softwares/laragon/www/signnsync/interpretation/feature_extracted"
base_output_folder = r"A:/Softwares/laragon/www/signnsync/interpretation/frames"

# Process face videos
face_input = os.path.normpath(os.path.join(base_input_folder, "face"))
face_output = os.path.normpath(os.path.join(base_output_folder, "face"))
if os.path.exists(face_input):
    extract_sharpened_frames(face_input, face_output)

# Process left-hand videos
left_hand_input = os.path.normpath(os.path.join(base_input_folder, "left_hand"))
left_hand_output = os.path.normpath(os.path.join(base_output_folder, "left_hand"))
if os.path.exists(left_hand_input):
    extract_sharpened_frames(left_hand_input, left_hand_output)

# Process right-hand videos
right_hand_input = os.path.normpath(os.path.join(base_input_folder, "right_hand"))
right_hand_output = os.path.normpath(os.path.join(base_output_folder, "right_hand"))
if os.path.exists(right_hand_input):
    extract_sharpened_frames(right_hand_input, right_hand_output)

print("✅ All feature-extracted videos converted to sharpened frames successfully!")

# 🚀 Call preprocess_image.py to preprocess extracted frames
print("🔄 Calling preprocess_image.py for preprocessing...")
subprocess.run([sys.executable, PREPROCESS_SCRIPT_PATH])
print("✅ Preprocessing completed successfully!")
