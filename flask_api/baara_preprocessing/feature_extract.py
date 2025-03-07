import cv2
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic

def extract_features(input_folder, output_folder):
    """
    Extracts face, left hand, and right hand videos from the input videos.

    Args:
        input_folder (str): Path to the folder containing raw videos.
        output_folder (str): Path to the folder where extracted feature videos will be saved.
    """

    # Create output directories
    face_folder = os.path.join(output_folder, "face")
    right_hand_folder = os.path.join(output_folder, "right")
    left_hand_folder = os.path.join(output_folder, "left")
    
    os.makedirs(face_folder, exist_ok=True)
    os.makedirs(right_hand_folder, exist_ok=True)
    os.makedirs(left_hand_folder, exist_ok=True)

    for video_file in os.listdir(input_folder):
        if not video_file.endswith(".mp4"):
            continue  # Skip non-video files

        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        face_output = cv2.VideoWriter(os.path.join(face_folder, f"face_{video_file}"),
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        right_hand_output = cv2.VideoWriter(os.path.join(right_hand_folder, f"right_hand_{video_file}"),
                                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        left_hand_output = cv2.VideoWriter(os.path.join(left_hand_folder, f"left_hand_{video_file}"),
                                           cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                h, w, _ = frame.shape
                padding = 30

                # Face Extraction
                if results.face_landmarks:
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in results.face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, x_max = min(x_min, x), max(x_max, x)
                        y_min, y_max = min(y_min, y), max(y_max, y)
                    x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                    y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size > 0:
                        face_output.write(cv2.resize(face_crop, (frame_width, frame_height)))

                # Right Hand Extraction
                if results.right_hand_landmarks:
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in results.right_hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, x_max = min(x_min, x), max(x_max, x)
                        y_min, y_max = min(y_min, y), max(y_max, y)
                    x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                    y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                    right_hand_crop = frame[y_min:y_max, x_min:x_max]
                    if right_hand_crop.size > 0:
                        right_hand_output.write(cv2.resize(right_hand_crop, (frame_width, frame_height)))

                # Left Hand Extraction
                if results.left_hand_landmarks:
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in results.left_hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, x_max = min(x_min, x), max(x_max, x)
                        y_min, y_max = min(y_min, y), max(y_max, y)
                    x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                    y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                    left_hand_crop = frame[y_min:y_max, x_min:x_max]
                    if left_hand_crop.size > 0:
                        left_hand_output.write(cv2.resize(left_hand_crop, (frame_width, frame_height)))

        cap.release()
        face_output.release()
        right_hand_output.release()
        left_hand_output.release()

        print(f"Processed: {video_file}")

    print("All videos processed successfully!")
