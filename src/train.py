
from pathlib import Path
import tensorflow as tf

from src.data_preprocessing import load_datasets
from src.model import build_model


# Project paths
PROJECT_PATH = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_PATH / "dataset" / "raw"
MODEL_DIR = PROJECT_PATH / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
(
    train_dataset,
    validation_dataset,
    test_dataset,
    class_names,
    num_classes
) = load_datasets(
    dataset_path=DATASET_PATH,
    image_size=(160, 160),
    batch_size=32
)


# Build model
model = build_model(
    num_classes=num_classes,
    image_size=(160, 160)
)


# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# Train model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[early_stop]
)


# Save model
model_path = MODEL_DIR / "tomato_disease_mobilenetv2.keras"

model.save(model_path)

print("Training completed!")
print("Model saved at:", model_path)
