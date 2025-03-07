import cv2
import numpy as np

def preprocess_image(image):
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))  # Adjust size to model input
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    return image
