from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
DATASET_DIR = PROJECT_ROOT / "data" / "chest_xray"


splits = [
    "train",
    "val",
    "test"
]

classes = [
    "NORMAL",
    "PNEUMONIA"
]


print("Dataset path:")
print(DATASET_DIR)


for split in splits:

    print(f"\n{split.upper()} DATASET")

    split_path = DATASET_DIR / split

    for class_name in classes:

        class_path = split_path / class_name

        if class_path.exists():

            images = [

                file
                for file in class_path.iterdir()

                if file.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            ]

            print(
                f"{class_name}: "
                f"{len(images)} images"
            )

        else:

            print(
                f"{class_name}: Folder not found"
            )