
from pathlib import Path
import tensorflow as tf
import numpy as np

# Project root directory
PROJECT_PATH = Path(__file__).resolve().parent.parent

# Trained model path
MODEL_PATH = (
    PROJECT_PATH
    / "models"
    / "tomato_disease_mobilenetv2_finetuned.keras"
)

# Dataset path
DATASET_PATH = PROJECT_PATH / "dataset" / "raw"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names
class_names = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])


def predict_disease(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    disease = class_names[predicted_index]

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    return disease, confidence
