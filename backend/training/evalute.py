import sys
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PYTHON PATH CONFIGURATION
# ============================================================

# Current file:
# backend/training/evaluate.py
#
# parents[0] = backend/training
# parents[1] = backend

BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BACKEND_DIR)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model.model import create_model

from training.dataset import create_dataloaders


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = BACKEND_DIR.parent


# ============================================================
# TRAINED MODEL PATH
# ============================================================

MODEL_PATH = (

    PROJECT_ROOT

    / "backend"

    / "model"

    / "trained_model.pth"

)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model():

    print(
        "\n========================================"
    )

    print(
        "       MODEL EVALUATION"
    )

    print(
        "       EfficientNet-B0"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print(

        "\nUsing device:",

        DEVICE

    )


    # --------------------------------------------------------
    # CHECK MODEL FILE
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(

            "\nERROR: Trained model not found!"

        )

        print(

            "Expected path:",

            MODEL_PATH

        )

        return


    print(

        "\nLoading trained model..."

    )


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    (

        train_loader,

        val_loader,

        test_loader,

        class_names

    ) = create_dataloaders()


    print(

        "\nClasses:",

        class_names

    )


    print(

        "Test images:",

        len(

            test_loader.dataset

        )

    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = create_model(

        num_classes=NUM_CLASSES,

        freeze_backbone=True

    )


    # --------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=DEVICE

        )

    )


    # Move model to device

    model = model.to(

        DEVICE

    )


    # Set evaluation mode

    model.eval()


    print(

        "\nTrained model loaded successfully!"

    )


    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------

    test_loss = 0.0


    all_labels = []


    all_predictions = []


    # ========================================================
    # TESTING
    # ========================================================

    print(

        "\nEvaluating test dataset..."

    )


    with torch.no_grad():


        for images, labels in test_loader:


            # Move images to device

            images = images.to(

                DEVICE

            )


            # Move labels to device

            labels = labels.to(

                DEVICE

            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            outputs = model(

                images

            )


            # ------------------------------------------------
            # CALCULATE LOSS
            # ------------------------------------------------

            loss = criterion(

                outputs,

                labels

            )


            test_loss += (

                loss.item()

            )


            # ------------------------------------------------
            # GET PREDICTED CLASS
            # ------------------------------------------------

            _, predictions = torch.max(

                outputs,

                1

            )


            # ------------------------------------------------
            # STORE LABELS
            # ------------------------------------------------

            all_labels.extend(

                labels.cpu().numpy()

            )


            # ------------------------------------------------
            # STORE PREDICTIONS
            # ------------------------------------------------

            all_predictions.extend(

                predictions.cpu().numpy()

            )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    # Average test loss

    average_test_loss = (

        test_loss

        / len(

            test_loader

        )

    )


    # Accuracy

    accuracy = accuracy_score(

        all_labels,

        all_predictions

    )


    # Precision

    precision = precision_score(

        all_labels,

        all_predictions,

        average="binary",

        zero_division=0

    )


    # Recall

    recall = recall_score(

        all_labels,

        all_predictions,

        average="binary",

        zero_division=0

    )


    # F1 Score

    f1 = f1_score(

        all_labels,

        all_predictions,

        average="binary",

        zero_division=0

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        all_labels,

        all_predictions

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(

        "\n========================================"

    )

    print(

        "       TEST RESULTS"

    )

    print(

        "========================================"

    )


    print(

        f"\nTest Loss: "
        f"{average_test_loss:.4f}"

    )


    print(

        f"Accuracy: "
        f"{accuracy * 100:.2f}%"

    )


    print(

        f"Precision: "
        f"{precision * 100:.2f}%"

    )


    print(

        f"Recall: "
        f"{recall * 100:.2f}%"

    )


    print(

        f"F1 Score: "
        f"{f1 * 100:.2f}%"

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(

        "\n========================================"

    )

    print(

        "       CONFUSION MATRIX"

    )

    print(

        "========================================"

    )


    print(

        "\nClass order:",

        class_names

    )


    print(

        "\n",

        cm

    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(

        "\n========================================"

    )

    print(

        "       CLASSIFICATION REPORT"

    )

    print(

        "========================================"

    )


    print(

        classification_report(

            all_labels,

            all_predictions,

            target_names=class_names,

            zero_division=0

        )

    )


    # ========================================================
    # CONFUSION MATRIX INTERPRETATION
    # ========================================================

    if cm.shape == (2, 2):

        tn = cm[0][0]

        fp = cm[0][1]

        fn = cm[1][0]

        tp = cm[1][1]


        print(

            "\n========================================"

        )

        print(

            "       CONFUSION MATRIX DETAILS"

        )

        print(

            "========================================"

        )


        print(

            "\nTrue Negative (TN):",

            tn

        )


        print(

            "False Positive (FP):",

            fp

        )


        print(

            "False Negative (FN):",

            fn

        )


        print(

            "True Positive (TP):",

            tp

        )


        print(

            "\nMedical interpretation:"

        )


        print(

            "False Negative = Pneumonia image "
            "predicted as NORMAL"

        )


        print(

            "False Positive = NORMAL image "
            "predicted as PNEUMONIA"

        )


    # ========================================================
    # EVALUATION COMPLETED
    # ========================================================

    print(

        "\n========================================"

    )

    print(

        "       EVALUATION COMPLETED"

    )

    print(

        "========================================"

    )


# ============================================================
# RUN EVALUATION
# ============================================================

if __name__ == "__main__":

    evaluate_model()