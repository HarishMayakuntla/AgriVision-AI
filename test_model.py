
from src.model import build_model

num_classes = 10

model = build_model(
    num_classes=num_classes,
    image_size=(160, 160)
)

model.summary()
