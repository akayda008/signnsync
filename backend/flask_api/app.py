from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load ML Models
sign_model = tf.keras.models.load_model("models/sign_language_model.h5")
emotion_model = tf.keras.models.load_model("models/emotion_model.h5")

# Function to process incoming image
def preprocess_image(image_data):
    try:
        # Decode base64 string to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array (RGB)
        image = np.array(image)

        # Resize to match model input
        image = cv2.resize(image, (64, 64))  # Ensure model input shape
        image = image.astype('float32') / 255.0  # Normalize

        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'video_frame' not in data:
            return jsonify({"error": "Missing 'video_frame' in request"}), 400

        # Preprocess image
        video_frame = preprocess_image(data['video_frame'])

        # Predict sign language
        sign_prediction = sign_model.predict(video_frame)
        sign_label = int(np.argmax(sign_prediction))  # Convert NumPy int to Python int

        # Predict emotion
        emotion_prediction = emotion_model.predict(video_frame)
        emotion_label = int(np.argmax(emotion_prediction))  

        return jsonify({
            "sign_language": sign_label,
            "emotion": emotion_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
