
from pathlib import Path

import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import load_datasets


# Project paths
PROJECT_PATH = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_PATH / "dataset" / "raw"
MODEL_PATH = PROJECT_PATH / "models" / "tomato_disease_mobilenetv2.keras"


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


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


# Evaluate model
test_loss, test_accuracy = model.evaluate(test_dataset)

print("Test Loss:", test_loss)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Generate predictions
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Tomato Disease Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
