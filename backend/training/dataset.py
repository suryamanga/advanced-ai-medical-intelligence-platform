from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "chest_xray"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# ============================================================
# IMAGENET NORMALIZATION
# EfficientNet pretrained weights use ImageNet normalization
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# TRAINING TRANSFORM
# Augmentation is applied ONLY to training images
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# No random augmentation
# ============================================================

val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    train_dir = DATASET_DIR / "train"

    test_dir = DATASET_DIR / "test"


    # --------------------------------------------------------
    # Create two versions of the training dataset
    #
    # One uses augmentation for training.
    # One does NOT use augmentation for validation.
    # --------------------------------------------------------

    train_dataset_augmented = datasets.ImageFolder(

        root=train_dir,

        transform=train_transform

    )


    train_dataset_no_augmentation = datasets.ImageFolder(

        root=train_dir,

        transform=val_test_transform

    )


    # --------------------------------------------------------
    # Create reproducible train/validation split
    # --------------------------------------------------------

    total_size = len(
        train_dataset_augmented
    )


    validation_size = int(

        total_size
        * VALIDATION_RATIO

    )


    training_size = (

        total_size
        - validation_size

    )


    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )


    # Generate random indices
    indices = torch.randperm(

        total_size,

        generator=generator

    ).tolist()


    train_indices = indices[
        :training_size
    ]


    val_indices = indices[
        training_size:
    ]


    # --------------------------------------------------------
    # Training subset
    # Uses augmentation
    # --------------------------------------------------------

    train_dataset = Subset(

        train_dataset_augmented,

        train_indices

    )


    # --------------------------------------------------------
    # Validation subset
    # Uses NO augmentation
    # --------------------------------------------------------

    val_dataset = Subset(

        train_dataset_no_augmentation,

        val_indices

    )


    # --------------------------------------------------------
    # Test dataset
    # Uses NO augmentation
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(

        root=test_dir,

        transform=val_test_transform

    )


    return (

        train_dataset,

        val_dataset,

        test_dataset,

        train_dataset_augmented.classes

    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    (

        train_dataset,

        val_dataset,

        test_dataset,

        class_names

    ) = load_datasets()


    # --------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS

    )


    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )


    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )


    return (

        train_loader,

        val_loader,

        test_loader,

        class_names

    )


# ============================================================
# TEST DATASET PIPELINE
# ============================================================

if __name__ == "__main__":

    (

        train_loader,

        val_loader,

        test_loader,

        class_names

    ) = create_dataloaders()


    print(
        "\nDataset loaded successfully!"
    )


    print(
        "\nDataset path:"
    )

    print(
        DATASET_DIR
    )


    print(
        "\nClasses:"
    )

    print(
        class_names
    )


    print(
        "\nTraining images:"
    )

    print(
        len(
            train_loader.dataset
        )
    )


    print(
        "\nValidation images:"
    )

    print(
        len(
            val_loader.dataset
        )
    )


    print(
        "\nTest images:"
    )

    print(
        len(
            test_loader.dataset
        )
    )


    print(
        "\nTraining batches:"
    )

    print(
        len(
            train_loader
        )
    )


    print(
        "\nValidation batches:"
    )

    print(
        len(
            val_loader
        )
    )


    print(
        "\nTest batches:"
    )

    print(
        len(
            test_loader
        )
    )