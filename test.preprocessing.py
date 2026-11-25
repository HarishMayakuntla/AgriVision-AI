
from pathlib import Path

from src.data_preprocessing import load_datasets


PROJECT_PATH = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_PATH / "dataset" / "raw"


train_dataset, validation_dataset, test_dataset, class_names, num_classes = load_datasets(
    dataset_path=DATASET_PATH,
    image_size=(160, 160),
    batch_size=32
)


print("Dataset loaded successfully!")

print("Number of classes:", num_classes)
print("Class names:", class_names)

print("Training batches:",
      len(train_dataset))

print("Validation batches:",
      len(validation_dataset))

print("Testing batches:",
      len(test_dataset))
