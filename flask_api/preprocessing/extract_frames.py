import cv2

def extract_frames(video_path, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened correctly
    if not cap.isOpened():
        print("❌ Error opening video file!")
        return []

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {len(frames)} frames.")
    return frames
