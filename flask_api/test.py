import cv2
import os

# Path to extracted face video
face_video_path = "A:/Softwares/laragon/www/signnsync/interpretation/feature_extracted/face/face_test.mp4"

if os.path.exists(face_video_path):
    print(f"[DEBUG] Face video exists: {face_video_path}")

    cap = cv2.VideoCapture(face_video_path)
    if not cap.isOpened():
        print("[ERROR] OpenCV failed to open extracted video!")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()
    print(f"[DEBUG] Extracted face video has {frame_count} frames.")
else:
    print("[ERROR] Face video file is missing!")
