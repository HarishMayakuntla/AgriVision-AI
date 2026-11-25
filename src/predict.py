
from pathlib import Path

import tensorflow as tf
import numpy as np


# Project paths
PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH / "models" / "tomato_disease_mobilenetv2_finetuned.keras"

DATASET_PATH = PROJECT_PATH / "dataset" / "raw"


# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Load class names
class_names = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])


def predict_disease(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    # Convert image to array
    img_array = tf.keras.utils.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    # Prediction
    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence



if __name__ == "__main__":

    # Find an existing tomato image
    image_files = list(
        DATASET_PATH.rglob("*.JPG")
    ) + list(
        DATASET_PATH.rglob("*.jpg")
    ) + list(
        DATASET_PATH.rglob("*.jpeg")
    ) + list(
        DATASET_PATH.rglob("*.png")
    )

    if not image_files:
        raise FileNotFoundError("No images found in dataset!")

    image_path = image_files[0]

    print("Testing Image:", image_path)

    disease, confidence = predict_disease(image_path)

    print("\nPredicted Disease:", disease)
    print(f"Confidence: {confidence:.2f}%")

