import os
import tensorflow as tf

# Get the absolute path to the 'model' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")

# Paths to model files
emotion_model_path = os.path.join(MODEL_PATH, "emotion_model.h5")
sign_model_path = os.path.join(MODEL_PATH, "sign_language_model.h5")

# ========================
# ✅ GPU Memory Handling
# ========================
# Prevent TensorFlow from allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Allocate memory dynamically
        print("[INFO] GPU detected. Memory growth enabled.")
    except RuntimeError as e:
        print(f"[WARNING] GPU memory configuration failed: {e}")
else:
    print("[INFO] No GPU found. Running on CPU.")

# ========================
# ✅ Check if model files exist
# ========================
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"[ERROR] Emotion model not found at {emotion_model_path}")

if not os.path.exists(sign_model_path):
    raise FileNotFoundError(f"[ERROR] Sign language model not found at {sign_model_path}")

# ========================
# ✅ Load the models safely
# ========================
try:
    emotion_model = tf.keras.models.load_model(emotion_model_path, compile=False)
    print("[INFO] Emotion model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to load emotion model: {str(e)}")

try:
    sign_model = tf.keras.models.load_model(sign_model_path, compile=False)
    print("[INFO] Sign language model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to load sign language model: {str(e)}")
