import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.app.ml.inference import classifier


router = APIRouter(tags=["image"])


@router.post("/image/predict")
async def predict_image(file: UploadFile = File(...)):

    # Validate file type
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=415,
            detail="Upload an image file.",
        )

    # Read uploaded image
    raw = await file.read()

    # Validate file size
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 12 MB limit.",
        )

    # Decode image
    try:
        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Could not decode image.",
        )

    # Prediction
    try:
        result = classifier.predict(image)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    # Grad-CAM
    try:
        gradcam = classifier.generate_gradcam(
            image=image,
            target_class=result["class_index"],
        )

        result["gradcam_png_base64"] = gradcam

    except Exception as exc:

        result["gradcam_note"] = (
            f"Grad-CAM unavailable: {type(exc).__name__}"
        )

    # Disclaimer
    result["disclaimer"] = (
        "Educational model output only—not a diagnosis. "
        "Do not use for clinical decisions."
    )

    return result