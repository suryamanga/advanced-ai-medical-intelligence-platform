import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 2


# ============================================================
# CREATE EFFICIENTNET-B0 MODEL
# ============================================================

def create_model(
    num_classes=NUM_CLASSES,
    freeze_backbone=True
):

    # Load EfficientNet-B0 pretrained on ImageNet
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )


    # --------------------------------------------------------
    # Freeze pretrained backbone
    # --------------------------------------------------------

    if freeze_backbone:

        for param in model.parameters():

            param.requires_grad = False


    # --------------------------------------------------------
    # Replace the original classifier
    #
    # Original:
    # ImageNet → 1000 classes
    #
    # Our project:
    # Chest X-ray → 2 classes
    #
    # NORMAL
    # PNEUMONIA
    # --------------------------------------------------------

    in_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(

        in_features,

        num_classes

    )


    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = create_model()


    print(
        "\nEfficientNet-B0 model created successfully!"
    )


    print(
        "\nNumber of output classes:",
        NUM_CLASSES
    )


    print(
        "\nClassifier:"
    )


    print(
        model.classifier
    )


    # Count trainable parameters

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


    total_params = sum(

        p.numel()

        for p in model.parameters()

    )


    print(
        "\nTotal parameters:",
        total_params
    )


    print(
        "Trainable parameters:",
        trainable_params
    )