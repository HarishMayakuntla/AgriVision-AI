
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import tempfile
import os
import io

from app.predictor import predict_disease

app = FastAPI(
    title="AgriVision-AI API",
    description="Tomato Plant Disease Detection API",
    version="1.1.0"
)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@app.get("/")
def home():
    return {
        "message": "Welcome to AgriVision-AI API",
        "status": "Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be less than 5 MB"
        )

    # Validate actual image contents
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.format not in ["JPEG", "PNG"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image format"
                )
            image.verify()

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image"
        )

    temp_path = None

    try:
        suffix = ".png" if file.content_type == "image/png" else ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        disease, confidence = predict_disease(temp_path)

        return {
            "filename": file.filename,
            "predicted_disease": disease,
            "confidence": round(confidence, 2),
            "status": "Prediction successful"
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during disease prediction"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
