import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Holistic model once
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def extract_features_from_frame(frame):
    if frame is None:
        return None, None, None

    h, w, _ = frame.shape
    padding = 30

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image)

    def crop_landmarks(landmarks):
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
        return frame[max(0, y_min - padding):min(h, y_max + padding),
                     max(0, x_min - padding):min(w, x_max + padding)] if y_max > y_min and x_max > x_min else None

    return (
        crop_landmarks(results.face_landmarks) if results.face_landmarks else None,
        crop_landmarks(results.right_hand_landmarks) if results.right_hand_landmarks else None,
        crop_landmarks(results.left_hand_landmarks) if results.left_hand_landmarks else None,
    )
