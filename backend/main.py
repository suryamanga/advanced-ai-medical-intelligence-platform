from pathlib import Path
import sys
import shutil
import uuid

import torch
import torch.nn.functional as F

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles


# ============================================================
# BACKEND PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent


# ============================================================
# ADD BACKEND TO PYTHON PATH
# ============================================================

if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from model.model import create_model

from xai.gradcam import generate_gradcam

from llm.report_generator import generate_medical_report

from database.database import (
    save_prediction,
    get_prediction_history,
    get_prediction_by_id
)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Advanced AI Medical Intelligence Platform",

    description=(
        "AI-powered medical image analysis "
        "using EfficientNet-B0, Grad-CAM and "
        "AI-assisted medical report generation."
    ),

    version="1.0.0"

)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = (

    BACKEND_DIR
    / "uploads"

)

OUTPUT_DIR = (

    BACKEND_DIR
    / "outputs"
    / "gradcam"

)


UPLOAD_DIR.mkdir(

    parents=True,

    exist_ok=True

)


OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================
# SERVE OUTPUT FILES
# ============================================================

app.mount(

    "/outputs",

    StaticFiles(

        directory=str(

            OUTPUT_DIR

        )

    ),

    name="outputs"

)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLASS_NAMES = [

    "NORMAL",

    "PNEUMONIA"

]


DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


MODEL_PATH = (

    BACKEND_DIR
    / "model"
    / "trained_model.pth"

)


# ============================================================
# LOAD MODEL
# ============================================================

print()

print(

    "========================================"

)

print(

    "Loading EfficientNet-B0 model..."

)

print(

    "========================================"

)


model = create_model(

    num_classes=2,

    freeze_backbone=False

)


if not MODEL_PATH.exists():

    raise FileNotFoundError(

        f"Trained model not found at: "
        f"{MODEL_PATH}"

    )


model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )

)


model = model.to(

    DEVICE

)


model.eval()


print(

    "EfficientNet-B0 model loaded successfully!"

)


print(

    "Using device:",

    DEVICE

)


print()


# ============================================================
# ROOT API
# ============================================================

@app.get("/")
def root():

    return {

        "message":
        "Advanced AI Medical Intelligence Platform",

        "status":
        "API is running",

        "model":
        "EfficientNet-B0"

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
        "healthy",

        "model":
        "EfficientNet-B0",

        "device":
        str(DEVICE)

    }


# ============================================================
# PREDICT API
# ============================================================

@app.post("/predict")
async def predict(

    file: UploadFile = File(...)

):

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail="No file uploaded."

        )


    # --------------------------------------------------------
    # CHECK IMAGE EXTENSION
    # --------------------------------------------------------

    allowed_extensions = [

        ".jpg",

        ".jpeg",

        ".png"

    ]


    file_extension = (

        Path(

            file.filename

        ).suffix.lower()

    )


    if file_extension not in allowed_extensions:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid file format. "
                "Please upload JPG, JPEG or PNG."
            )

        )


    # --------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------------------------

    unique_filename = (

        f"{uuid.uuid4().hex}_"

        f"{file.filename}"

    )


    image_path = (

        UPLOAD_DIR

        / unique_filename

    )


    # --------------------------------------------------------
    # SAVE UPLOADED IMAGE
    # --------------------------------------------------------

    with open(

        image_path,

        "wb"

    ) as buffer:

        shutil.copyfileobj(

            file.file,

            buffer

        )


    print()

    print(

        "========================================"

    )

    print(

        "NEW IMAGE PREDICTION"

    )

    print(

        "========================================"

    )

    print(

        "Filename:",

        file.filename

    )


    # ========================================================
    # IMAGE PREDICTION
    # ========================================================

    try:

        from PIL import Image

        from torchvision import transforms


        # ----------------------------------------------------
        # IMAGE TRANSFORMATION
        # ----------------------------------------------------

        transform = transforms.Compose([

            transforms.Resize(

                (224, 224)

            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[

                    0.485,

                    0.456,

                    0.406

                ],

                std=[

                    0.229,

                    0.224,

                    0.225

                ]

            )

        ])


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(

            image_path

        ).convert(

            "RGB"

        )


        # ----------------------------------------------------
        # TRANSFORM IMAGE
        # ----------------------------------------------------

        image_tensor = transform(

            image

        ).unsqueeze(

            0

        )


        image_tensor = image_tensor.to(

            DEVICE

        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(

                image_tensor

            )


            probabilities = F.softmax(

                outputs,

                dim=1

            )


            confidence, predicted_class = (

                torch.max(

                    probabilities,

                    1

                )

            )


        prediction = CLASS_NAMES[

            predicted_class.item()

        ]


        confidence_value = (

            confidence.item()

        )


        confidence_percentage = (

            confidence_value * 100

        )


        print()

        print(

            "Prediction:",

            prediction

        )


        print(

            "Confidence:",

            f"{confidence_percentage:.2f}%"

        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(

                "Prediction failed: "

                + str(e)

            )

        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    heatmap_path = None

    overlay_path = None


    try:

        print()

        print(

            "Generating Grad-CAM..."

        )


        gradcam_result = generate_gradcam(

            image_path=str(

                image_path

            )

        )


        # ----------------------------------------------------
        # HANDLE GRAD-CAM RESULT
        # ----------------------------------------------------

        if isinstance(

            gradcam_result,

            dict

        ):

            heatmap_path = (

                gradcam_result.get(

                    "heatmap_path"

                )

            )


            overlay_path = (

                gradcam_result.get(

                    "overlay_path"

                )

            )


        elif isinstance(

            gradcam_result,

            tuple

        ):

            if len(

                gradcam_result

            ) >= 2:

                heatmap_path = str(

                    gradcam_result[0]

                )

                overlay_path = str(

                    gradcam_result[1]

                )


        print(

            "Grad-CAM generated successfully!"

        )


    except Exception as e:

        print(

            "Grad-CAM generation failed:",

            str(e)

        )


    # ========================================================
    # AI MEDICAL REPORT
    # ========================================================

    try:

        print()

        print(

            "Generating AI-assisted medical report..."

        )


        medical_report = (

            generate_medical_report(

                prediction,

                confidence_percentage

            )

        )


        print(

            "Medical report generated successfully!"

        )


    except Exception as e:

        print(

            "Medical report generation failed:",

            str(e)

        )


        medical_report = (

            "AI-assisted medical report "
            "could not be generated."

        )


    # ========================================================
    # SAVE PREDICTION TO DATABASE
    # ========================================================

    try:

        prediction_id = save_prediction(

            filename=file.filename,

            prediction=prediction,

            confidence=confidence_percentage,

            medical_report=medical_report,

            heatmap_path=(

                str(heatmap_path)

                if heatmap_path

                else None

            ),

            gradcam_path=(

                str(overlay_path)

                if overlay_path

                else None

            )

        )


        print()

        print(

            "Prediction saved to database."

        )


        print(

            "Prediction ID:",

            prediction_id

        )


    except Exception as e:

        print(

            "Database save failed:",

            str(e)

        )


        prediction_id = None


    # ========================================================
    # FINAL API RESPONSE
    # ========================================================

    print()

    print(

        "========================================"

    )

    print(

        "PREDICTION COMPLETED"

    )

    print(

        "========================================"

    )


    return {

        "success": True,

        "prediction_id":

            prediction_id,

        "filename":

            file.filename,

        "prediction":

            prediction,

        "confidence":

            round(

                confidence_percentage,

                2

            ),

        "gradcam":

            "Generated"

            if overlay_path

            else "Not Generated",

        "heatmap_path":

            str(

                heatmap_path

            )

            if heatmap_path

            else None,

        "overlay_path":

            str(

                overlay_path

            )

            if overlay_path

            else None,

        "medical_report":

            medical_report

    }


# ============================================================
# GET ALL PREDICTION HISTORY
# ============================================================

@app.get("/predictions")
def prediction_history():

    try:

        records = (

            get_prediction_history()

        )


        return {

            "success": True,

            "count": len(

                records

            ),

            "predictions": records

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(

                "Could not retrieve "
                "prediction history: "

                + str(e)

            )

        )


# ============================================================
# GET SINGLE PREDICTION
# ============================================================

@app.get(

    "/predictions/{prediction_id}"

)

def get_single_prediction(

    prediction_id: int

):

    try:

        record = (

            get_prediction_by_id(

                prediction_id

            )

        )


        if record is None:

            raise HTTPException(

                status_code=404,

                detail=(

                    "Prediction not found."

                )

            )


        return {

            "success": True,

            "prediction": record

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(

                "Could not retrieve "
                "prediction: "

                + str(e)

            )

        )


# ============================================================
# APPLICATION STARTUP
# ============================================================

@app.on_event(

    "startup"

)

async def startup_event():

    print()

    print(

        "========================================"

    )

    print(

        "Advanced AI Medical Intelligence Platform"

    )

    print(

        "API STARTED SUCCESSFULLY"

    )

    print(

        "========================================"

    )

    print(

        "Model: EfficientNet-B0"

    )

    print(

        "Device:",

        DEVICE

    )

    print()
