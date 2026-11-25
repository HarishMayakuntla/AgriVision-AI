
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_PATH
    / "models"
    / "tomato_disease_mobilenetv2_finetuned.keras"
)

print("Model Path:", MODEL_PATH)
print("Model Exists:", MODEL_PATH.exists())