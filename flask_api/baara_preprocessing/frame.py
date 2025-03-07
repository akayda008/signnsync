import cv2
import os

def extract_sharpened_frames(input_folder, output_folder, frame_rate=5):
    """
    Extracts frames from all videos in a given folder, applies sharpening, and saves them.

    Args:
        input_folder (str): Directory containing input video files.
        output_folder (str): Directory where extracted frames will be saved.
        frame_rate (int): Number of frames to extract per second.
    """

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for video_file in os.listdir(input_folder):
        if video_file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(input_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_folder = os.path.join(output_folder, video_name)
            
            os.makedirs(video_output_folder, exist_ok=True)  # Create subfolder for frames

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(1, fps // frame_rate)

            frame_count = 0
            saved_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  

                # Process only selected frames
                if frame_count % frame_interval == 0:
                    # Apply Sharpening to Reduce Motion Blur
                    blurred = cv2.GaussianBlur(frame, (5, 5), 0)  
                    sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

                    # Save sharpened frame
                    frame_filename = os.path.join(video_output_folder, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_filename, sharpened)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"Processed frames saved in: {video_output_folder}")

# Define input and output directories
base_input_folder = r"A:/Christ/Academics/CIA/CS Project/Data/ISL/Trust/Feature_Extract_trust"
base_output_folder = r"A:/Christ/Academics/CIA/CS Project/Data/ISL/Trust/Frames_trust"

# Process face videos
extract_sharpened_frames(os.path.join(base_input_folder, "trust_face"), os.path.join(base_output_folder, "frame_trust_face"))

# Process left-hand videos
extract_sharpened_frames(os.path.join(base_input_folder, "trust_left"), os.path.join(base_output_folder, "frame_trust_left"))

# Process right-hand videos
extract_sharpened_frames(os.path.join(base_input_folder, "trust_right"), os.path.join(base_output_folder, "frame_trust_right"))

print("All videos converted to frames successfully!")
