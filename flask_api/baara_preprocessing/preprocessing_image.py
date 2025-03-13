import cv2
import os
import numpy as np
import subprocess
import sys

# Paths
BASE_PATH = "A:/Softwares/laragon/www/signnsync/interpretation/"
FRAMES_PATH = os.path.normpath(os.path.join(BASE_PATH, "frames"))
PREPROCESSED_PATH = os.path.normpath(os.path.join(BASE_PATH, "preprocessed"))
ROUTES_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "routes.py")  # Path to routes.py

# Ensure output folders exist
for subfolder in ["face", "left_hand", "right_hand"]:
    os.makedirs(os.path.normpath(os.path.join(PREPROCESSED_PATH, subfolder)), exist_ok=True)

def preprocess_images(input_folder, output_folder, frame_size=(64, 64)):
    """
    Preprocesses extracted frames for model prediction:
    - Resizes frames to (64x64)
    - Normalizes pixel values (0 to 1)
    - Saves preprocessed images in the output folder.

    :param input_folder: Folder containing extracted frames (face, left_hand, right_hand)
    :param output_folder: Folder where preprocessed images will be saved
    :param frame_size: Target frame width & height (default: 64x64)
    """

    if not os.path.exists(input_folder) or not os.listdir(input_folder):
        print(f"⚠️ No frames found in: {input_folder}")
        return  # Skip if no images found

    processed_count = 0

    for image_file in sorted(os.listdir(input_folder)):  # Ensure sorted order
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            input_image_path = os.path.normpath(os.path.join(input_folder, image_file))
            output_image_path = os.path.normpath(os.path.join(output_folder, image_file))

            # Load image
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"⚠️ Skipping corrupted file: {input_image_path}")
                continue

            # Resize and normalize
            image = cv2.resize(image, frame_size)  # Resize to 64x64
            image = image.astype(np.float32) / 255.0  # Normalize pixel values (0-1)

            # Convert back to uint8 for saving
            image_uint8 = (image * 255).astype(np.uint8)
            cv2.imwrite(output_image_path, image_uint8)
            processed_count += 1

    print(f"✅ {processed_count} images preprocessed and saved in: {output_folder}")

# Process face images
preprocess_images(os.path.normpath(os.path.join(FRAMES_PATH, "face")), os.path.normpath(os.path.join(PREPROCESSED_PATH, "face")))

# Process left-hand images
preprocess_images(os.path.normpath(os.path.join(FRAMES_PATH, "left_hand")), os.path.normpath(os.path.join(PREPROCESSED_PATH, "left_hand")))

# Process right-hand images
preprocess_images(os.path.normpath(os.path.join(FRAMES_PATH, "right_hand")), os.path.normpath(os.path.join(PREPROCESSED_PATH, "right_hand")))

print("✅ All images preprocessed successfully!")