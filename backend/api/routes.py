from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from model.predict import predict_image
from xai.gradcam import generate_gradcam
from llm.report_generator import generate_medical_report
from database.database import save_prediction

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid medical image."
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Deep Learning Prediction
        prediction = predict_image(image)

        # Grad-CAM
        gradcam_image = generate_gradcam(image)

        # LLM Medical Report
        report = generate_medical_report(
            prediction=prediction
        )

        # Save Prediction History
        save_prediction(
            prediction=prediction,
            report=report
        )

        return {
            "prediction": prediction,
            "report": report,
            "gradcam": gradcam_image
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/history")
def get_prediction_history():

    from database.database import get_history

    history = get_history()

    return {
        "history": history
    }