import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# ============================================================
# PYTHON PATH CONFIGURATION
# ============================================================

# Current file:
# backend/xai/gradcam.py

# parents[0] = backend/xai
# parents[1] = backend
# parents[2] = project root

BACKEND_DIR = Path(__file__).resolve().parents[1]

PROJECT_ROOT = BACKEND_DIR.parent


# Add backend directory to Python path
if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )


# ============================================================
# PROJECT IMPORT
# ============================================================

from model.model import create_model


# ============================================================
# PATH CONFIGURATION
# ============================================================

MODEL_PATH = (

    BACKEND_DIR

    / "model"

    / "trained_model.pth"

)


# ============================================================
# GRAD-CAM OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (

    BACKEND_DIR

    / "xai"

    / "outputs"

)


# Create output directory

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2


# Dataset class order

CLASS_NAMES = [

    "NORMAL",

    "PNEUMONIA"

]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(

    [

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

    ]

)


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:


    def __init__(

        self,

        model,

        target_layer

    ):


        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None


        # ----------------------------------------------------
        # REGISTER FORWARD HOOK
        # ----------------------------------------------------

        self.forward_handle = (

            self.target_layer.register_forward_hook(

                self.save_activation

            )

        )


        # ----------------------------------------------------
        # REGISTER BACKWARD HOOK
        # ----------------------------------------------------

        self.backward_handle = (

            self.target_layer.register_full_backward_hook(

                self.save_gradient

            )

        )


    # ========================================================
    # SAVE FORWARD ACTIVATIONS
    # ========================================================

    def save_activation(

        self,

        module,

        input,

        output

    ):


        self.activations = output


    # ========================================================
    # SAVE BACKWARD GRADIENTS
    # ========================================================

    def save_gradient(

        self,

        module,

        grad_input,

        grad_output

    ):


        if (

            grad_output is not None

            and len(grad_output) > 0

        ):

            self.gradients = grad_output[0]


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    def generate(

        self,

        image_tensor,

        class_index

    ):


        # ----------------------------------------------------
        # CLEAR PREVIOUS DATA
        # ----------------------------------------------------

        self.activations = None

        self.gradients = None


        # ----------------------------------------------------
        # CLEAR MODEL GRADIENTS
        # ----------------------------------------------------

        self.model.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        output = self.model(

            image_tensor

        )


        # ----------------------------------------------------
        # SELECT TARGET CLASS
        # ----------------------------------------------------

        target = output[

            0,

            class_index

        ]


        # ----------------------------------------------------
        # BACKWARD PASS
        # ----------------------------------------------------

        target.backward()


        # ----------------------------------------------------
        # CHECK ACTIVATIONS
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(

                "Grad-CAM Error: "
                "Activations were not captured."

            )


        # ----------------------------------------------------
        # CHECK GRADIENTS
        # ----------------------------------------------------

        if self.gradients is None:

            raise RuntimeError(

                "Grad-CAM Error: "
                "Gradients were not captured."

            )


        # ----------------------------------------------------
        # GET ACTIVATIONS
        # ----------------------------------------------------

        activations = self.activations


        # ----------------------------------------------------
        # GET GRADIENTS
        # ----------------------------------------------------

        gradients = self.gradients


        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        weights = torch.mean(

            gradients,

            dim=(2, 3),

            keepdim=True

        )


        # ----------------------------------------------------
        # WEIGHTED ACTIVATIONS
        # ----------------------------------------------------

        cam = (

            weights

            * activations

        ).sum(

            dim=1,

            keepdim=True

        )


        # ----------------------------------------------------
        # APPLY RELU
        # ----------------------------------------------------

        cam = F.relu(

            cam

        )


        # ----------------------------------------------------
        # REMOVE BATCH DIMENSION
        # ----------------------------------------------------

        cam = cam.squeeze(

            0

        )


        # ----------------------------------------------------
        # DETACH FROM COMPUTATION GRAPH
        # ----------------------------------------------------

        cam = cam.detach()


        # ----------------------------------------------------
        # MOVE TO CPU
        # ----------------------------------------------------

        cam = cam.cpu()


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        cam = cam.numpy()


        # ----------------------------------------------------
        # REMOVE CHANNEL DIMENSION
        # ----------------------------------------------------

        cam = cam.squeeze(

            0

        )


        # ----------------------------------------------------
        # NORMALIZE CAM
        # ----------------------------------------------------

        cam = (

            cam

            - cam.min()

        )


        if cam.max() != 0:

            cam = (

                cam

                / cam.max()

            )


        return cam


    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    def remove_hooks(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():


    print(

        "\nLoading trained EfficientNet-B0..."

    )


    # --------------------------------------------------------
    # CREATE MODEL ARCHITECTURE
    # --------------------------------------------------------

    model = create_model(

        num_classes=NUM_CLASSES,

        freeze_backbone=True

    )


    # --------------------------------------------------------
    # ENABLE GRADIENTS
    #
    # Required for Grad-CAM
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = True


    # --------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=DEVICE

        )

    )


    # --------------------------------------------------------
    # MOVE MODEL TO DEVICE
    # --------------------------------------------------------

    model = model.to(

        DEVICE

    )


    # --------------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------------

    model.eval()


    print(

        "Model loaded successfully!"

    )


    return model


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(

    image_path

):


    print(

        "\nProcessing image:",

        image_path

    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original_image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    # --------------------------------------------------------
    # PREPARE IMAGE
    # --------------------------------------------------------

    image_tensor = transform(

        original_image

    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    image_tensor = image_tensor.unsqueeze(

        0

    )


    # --------------------------------------------------------
    # MOVE IMAGE TO DEVICE
    # --------------------------------------------------------

    image_tensor = image_tensor.to(

        DEVICE

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    # IMPORTANT:
    # Do NOT use torch.no_grad() here.
    #
    # Grad-CAM requires gradients.

    outputs = model(

        image_tensor

    )


    # --------------------------------------------------------
    # CALCULATE PROBABILITIES
    # --------------------------------------------------------

    probabilities = torch.softmax(

        outputs,

        dim=1

    )


    # --------------------------------------------------------
    # GET PREDICTED CLASS
    # --------------------------------------------------------

    predicted_class = torch.argmax(

        probabilities,

        dim=1

    ).item()


    # --------------------------------------------------------
    # GET CONFIDENCE
    # --------------------------------------------------------

    confidence = probabilities[

        0,

        predicted_class

    ].item()


    # --------------------------------------------------------
    # GET PREDICTION NAME
    # --------------------------------------------------------

    prediction = CLASS_NAMES[

        predicted_class

    ]


    # ========================================================
    # PRINT PREDICTION
    # ========================================================

    print(

        "\nPrediction:",

        prediction

    )


    print(

        "Confidence:",

        f"{confidence * 100:.2f}%"

    )


    # ========================================================
    # SELECT TARGET LAYER
    # ========================================================

    # EfficientNet-B0 final feature block
    #
    # model.features[-1][0]
    #
    # selects the final Conv2d layer.

    target_layer = (

        model.features[-1][0]

    )


    print(

        "\nTarget layer selected:",

        target_layer

    )


    # ========================================================
    # CREATE GRAD-CAM
    # ========================================================

    gradcam = GradCAM(

        model,

        target_layer

    )


    try:


        # ----------------------------------------------------
        # GENERATE CAM
        # ----------------------------------------------------

        cam = gradcam.generate(

            image_tensor,

            predicted_class

        )


    finally:


        # ----------------------------------------------------
        # REMOVE HOOKS
        # ----------------------------------------------------

        gradcam.remove_hooks()


    # ========================================================
    # LOAD ORIGINAL IMAGE USING OPENCV
    # ========================================================

    original = cv2.imread(

        str(

            image_path

        )

    )


    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if original is None:

        raise ValueError(

            "Unable to load image using OpenCV."

        )


    # --------------------------------------------------------
    # RESIZE ORIGINAL IMAGE
    # --------------------------------------------------------

    original = cv2.resize(

        original,

        (

            224,

            224

        )

    )


    # ========================================================
    # RESIZE CAM
    # ========================================================

    cam = cv2.resize(

        cam,

        (

            original.shape[1],

            original.shape[0]

        )

    )


    # ========================================================
    # CREATE HEATMAP
    # ========================================================

    heatmap = np.uint8(

        255

        * cam

    )


    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(

        original,

        0.6,

        heatmap,

        0.4,

        0

    )


    # ========================================================
    # GET IMAGE NAME
    # ========================================================

    image_name = Path(

        image_path

    ).stem


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    heatmap_path = (

        OUTPUT_DIR

        / f"{image_name}_heatmap.jpg"

    )


    overlay_path = (

        OUTPUT_DIR

        / f"{image_name}_gradcam.jpg"

    )


    # ========================================================
    # SAVE HEATMAP
    # ========================================================

    heatmap_saved = cv2.imwrite(

        str(

            heatmap_path

        ),

        heatmap

    )


    # ========================================================
    # SAVE GRAD-CAM OVERLAY
    # ========================================================

    overlay_saved = cv2.imwrite(

        str(

            overlay_path

        ),

        overlay

    )


    # --------------------------------------------------------
    # CHECK FILE SAVING
    # --------------------------------------------------------

    if not heatmap_saved:

        raise RuntimeError(

            "Failed to save Grad-CAM heatmap."

        )


    if not overlay_saved:

        raise RuntimeError(

            "Failed to save Grad-CAM overlay."

        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(

        "\n========================================"

    )


    print(

        "       GRAD-CAM COMPLETED"

    )


    print(

        "========================================"

    )


    print(

        "\nPrediction:",

        prediction

    )


    print(

        "Confidence:",

        f"{confidence * 100:.2f}%"

    )


    print(

        "\nHeatmap saved at:",

        heatmap_path

    )


    print(

        "Grad-CAM overlay saved at:",

        overlay_path

    )


    # ========================================================
    # RETURN RESULTS TO FASTAPI
    # ========================================================

    return {

        "prediction": prediction,

        "confidence": confidence,

        "heatmap_path": str(

            heatmap_path

        ),

        "overlay_path": str(

            overlay_path

        )

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # TEST IMAGE PATH
    # ========================================================

    image_path = (

        PROJECT_ROOT

        / "data"

        / "chest_xray"

        / "test"

        / "NORMAL"

        / "IM-0010-0001.jpeg"

    )


    # ========================================================
    # CHECK IMAGE EXISTS
    # ========================================================

    if not image_path.exists():


        print(

            "\nERROR: Image not found!"

        )


        print(

            "\nCurrent path:",

            image_path

        )


        print(

            "\nPlease update the image_path "
            "at the bottom of gradcam.py."

        )


    else:


        # ====================================================
        # RUN GRAD-CAM
        # ====================================================

        result = generate_gradcam(

            image_path

        )


        # ====================================================
        # PRINT RETURNED RESULT
        # ====================================================

        print(

            "\nReturned Result:"

        )


        print(

            result

        )