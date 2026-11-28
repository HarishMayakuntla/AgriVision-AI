
from pathlib import Path
from src.validator import validate_tomato_leaf

PROJECT_PATH = Path(__file__).resolve().parent

test_folder = (
    PROJECT_PATH
    / "dataset"
    / "binary_dataset"
)

for class_name in ["tomato_leaf", "not_tomato_leaf"]:

    folder = test_folder / class_name

    print("\nActual class:", class_name)

    for image_path in list(folder.glob("*.jpg"))[:5]:

        is_tomato, confidence = validate_tomato_leaf(image_path)

        predicted_class = (
            "Tomato" if is_tomato else "Non-Tomato"
        )

        print(
            image_path.name,
            "| Predicted:", predicted_class,
            "| Confidence:", round(confidence, 2), "%"
        )
