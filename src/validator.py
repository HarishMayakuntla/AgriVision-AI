
from pathlib import Path
import tensorflow as tf
import numpy as np

PROJECT_PATH = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_PATH
    / "models"
    / "tomato_leaf_validator_mobilenetv2.keras"
)

validator_model = tf.keras.models.load_model(MODEL_PATH)


def validate_tomato_leaf(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Model already contains MobileNetV2 preprocessing.
    # So, do not preprocess the image again.

    prediction = validator_model.predict(
        img_array,
        verbose=0
    )

    tomato_probability = float(prediction[0][0])

    if tomato_probability >= 0.5:
        return True, tomato_probability * 100

    return False, (1 - tomato_probability) * 100
