import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# PYTHON PATH CONFIGURATION
# ============================================================

# Get the backend directory
#
# Current file:
# backend/training/train.py
#
# parents[0] = backend/training
# parents[1] = backend
#
BACKEND_DIR = Path(__file__).resolve().parents[1]


# Add backend directory to Python path
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

# Get main project directory
#
# Advanced AI Medical Intelligence Platform/
#
PROJECT_ROOT = BACKEND_DIR.parent


# ============================================================
# MODEL SAVE PATH
# ============================================================

MODEL_SAVE_PATH = (

    PROJECT_ROOT

    / "backend"

    / "model"

    / "trained_model.pth"

)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of classes
#
# 0 = NORMAL
# 1 = PNEUMONIA
#
NUM_CLASSES = 2


# Number of training epochs
EPOCHS = 5


# Learning rate
LEARNING_RATE = 0.001


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

# Use GPU if available
# Otherwise use CPU

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():

    print(
        "\n========================================"
    )

    print(
        "      AI MEDICAL IMAGE CLASSIFICATION"
    )

    print(
        "      EfficientNet-B0 Training"
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
    # LOAD DATASETS
    # --------------------------------------------------------

    print(

        "\nLoading datasets..."

    )


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

        "Training images:",

        len(
            train_loader.dataset
        )

    )


    print(

        "Validation images:",

        len(
            val_loader.dataset
        )

    )


    print(

        "Test images:",

        len(
            test_loader.dataset
        )

    )


    # --------------------------------------------------------
    # CREATE EFFICIENTNET-B0 MODEL
    # --------------------------------------------------------

    print(

        "\nCreating EfficientNet-B0 model..."

    )


    model = create_model(

        num_classes=NUM_CLASSES,

        freeze_backbone=True

    )


    # Move model to device

    model = model.to(

        DEVICE

    )


    print(

        "EfficientNet-B0 model created successfully!"

    )


    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # OPTIMIZER
    #
    # Only the classifier is trainable because
    # the EfficientNet backbone is frozen.
    # --------------------------------------------------------

    optimizer = optim.Adam(

        model.classifier.parameters(),

        lr=LEARNING_RATE

    )


    # --------------------------------------------------------
    # TRACK BEST MODEL
    # --------------------------------------------------------

    best_val_accuracy = 0.0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(

        EPOCHS

    ):


        # ====================================================
        # TRAINING
        # ====================================================

        model.train()


        # Track total training loss

        running_loss = 0.0


        # Correct predictions

        correct = 0


        # Total samples

        total = 0


        # ----------------------------------------------------
        # PROCESS TRAINING BATCHES
        # ----------------------------------------------------

        for images, labels in train_loader:


            # Move images to CPU/GPU

            images = images.to(

                DEVICE

            )


            # Move labels to CPU/GPU

            labels = labels.to(

                DEVICE

            )


            # ------------------------------------------------
            # CLEAR PREVIOUS GRADIENTS
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # FORWARD PASS
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


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # UPDATE MODEL PARAMETERS
            # ------------------------------------------------

            optimizer.step()


            # ------------------------------------------------
            # TRACK LOSS
            # ------------------------------------------------

            running_loss += (

                loss.item()

            )


            # ------------------------------------------------
            # GET PREDICTIONS
            # ------------------------------------------------

            _, predicted = torch.max(

                outputs,

                1

            )


            # ------------------------------------------------
            # UPDATE TOTAL SAMPLES
            # ------------------------------------------------

            total += labels.size(

                0

            )


            # ------------------------------------------------
            # UPDATE CORRECT PREDICTIONS
            # ------------------------------------------------

            correct += (

                predicted == labels

            ).sum().item()


        # ----------------------------------------------------
        # CALCULATE TRAINING ACCURACY
        # ----------------------------------------------------

        train_accuracy = (

            100

            * correct

            / total

        )


        # ----------------------------------------------------
        # CALCULATE TRAINING LOSS
        # ----------------------------------------------------

        train_loss = (

            running_loss

            / len(

                train_loader

            )

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        # Validation correct predictions

        val_correct = 0


        # Validation total samples

        val_total = 0


        # Validation loss

        val_loss = 0.0


        # Disable gradient calculation
        # This saves memory and computation

        with torch.no_grad():


            # ------------------------------------------------
            # PROCESS VALIDATION BATCHES
            # ------------------------------------------------

            for images, labels in val_loader:


                # Move images to device

                images = images.to(

                    DEVICE

                )


                # Move labels to device

                labels = labels.to(

                    DEVICE

                )


                # ------------------------------------------------
                # FORWARD PASS
                # ------------------------------------------------

                outputs = model(

                    images

                )


                # ------------------------------------------------
                # CALCULATE VALIDATION LOSS
                # ------------------------------------------------

                loss = criterion(

                    outputs,

                    labels

                )


                # Add loss

                val_loss += (

                    loss.item()

                )


                # ------------------------------------------------
                # GET PREDICTIONS
                # ------------------------------------------------

                _, predicted = torch.max(

                    outputs,

                    1

                )


                # ------------------------------------------------
                # UPDATE TOTAL
                # ------------------------------------------------

                val_total += labels.size(

                    0

                )


                # ------------------------------------------------
                # UPDATE CORRECT
                # ------------------------------------------------

                val_correct += (

                    predicted == labels

                ).sum().item()


        # ----------------------------------------------------
        # CALCULATE VALIDATION ACCURACY
        # ----------------------------------------------------

        val_accuracy = (

            100

            * val_correct

            / val_total

        )


        # ----------------------------------------------------
        # CALCULATE VALIDATION LOSS
        # ----------------------------------------------------

        val_loss = (

            val_loss

            / len(

                val_loader

            )

        )


        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print(

            "\n----------------------------------------"

        )


        print(

            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"

        )


        print(

            f"Training Loss: "
            f"{train_loss:.4f}"

        )


        print(

            f"Training Accuracy: "
            f"{train_accuracy:.2f}%"

        )


        print(

            f"Validation Loss: "
            f"{val_loss:.4f}"

        )


        print(

            f"Validation Accuracy: "
            f"{val_accuracy:.2f}%"

        )


        print(

            "----------------------------------------"

        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (

            val_accuracy

            > best_val_accuracy

        ):


            # Update best accuracy

            best_val_accuracy = (

                val_accuracy

            )


            # Save model weights

            torch.save(

                model.state_dict(),

                MODEL_SAVE_PATH

            )


            print(

                "\nBest model saved!"

            )


            print(

                "Saved at:",

                MODEL_SAVE_PATH

            )


    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print(

        "\n========================================"

    )


    print(

        "       TRAINING COMPLETED"

    )


    print(

        "========================================"

    )


    print(

        "\nBest Validation Accuracy:",

        f"{best_val_accuracy:.2f}%"

    )


    print(

        "\nModel saved at:",

        MODEL_SAVE_PATH

    )


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_model()