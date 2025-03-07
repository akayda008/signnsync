import cv2
import os
import numpy as np

def preprocess_images(input_folder, output_folder, frame_size=(128, 128)):
    """
    Preprocesses images for LSTM/RNN model training.
    - Resizes frames to a fixed dimension.
    - Normalizes pixel values.
    - Saves preprocessed images in the output folder.

    Args:
        input_folder (str): Folder containing extracted frames.
        output_folder (str): Folder where preprocessed images will be saved.
        frame_size (tuple): Target frame width & height (default: 128x128).
    """

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Only process subdirectories (e.g., face_1, face_2, ...)
            output_subfolder = os.path.join(output_folder, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)

            for image_file in os.listdir(subfolder_path):
                if image_file.endswith((".png", ".jpg", ".jpeg")):
                    input_image_path = os.path.join(subfolder_path, image_file)
                    output_image_path = os.path.join(output_subfolder, image_file)

                    # Load, resize, and normalize image
                    image = cv2.imread(input_image_path)
                    if image is None:
                        print(f"Skipping corrupted file: {input_image_path}")
                        continue

                    image = cv2.resize(image, frame_size)  # Resize
                    image = image / 255.0  # Normalize pixel values

                    # Convert back to uint8 for saving
                    image_uint8 = (image * 255).astype(np.uint8)
                    cv2.imwrite(output_image_path, image_uint8)

            print(f"Preprocessed images saved in: {output_subfolder}")

def run_image_preprocessing():
    """
    Runs image preprocessing for extracted frames of face, left hand, and right hand.
    """
    base_input_folder = r"A:/Christ/Academics/CIA/CS Project/Data/ISL/Trust/Frames_trust"
    base_output_folder = r"A:/Christ/Academics/CIA/CS Project/Data/ISL/Trust/Preprocessed_trust"

    preprocess_images(os.path.join(base_input_folder, "frame_trust_face"), os.path.join(base_output_folder, "preprocessed_trust_face"))
    preprocess_images(os.path.join(base_input_folder, "frame_trust_left"), os.path.join(base_output_folder, "preprocessed_trust_left"))
    preprocess_images(os.path.join(base_input_folder, "frame_trust_right"), os.path.join(base_output_folder, "preprocessed_trust_right"))

    print("All images preprocessed successfully!")
