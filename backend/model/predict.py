import torch
from torchvision import transforms
from PIL import Image

from model.model import create_model


# Change this according to your dataset classes
CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Create model
model = create_model(
    num_classes=len(CLASS_NAMES)
)


# Load trained model
MODEL_PATH = "model/trained_model.pth"

try:
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )
except FileNotFoundError:
    print(
        "Warning: trained_model.pth not found. "
        "Train the model before making predictions."
    )


model.to(DEVICE)
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image: Image.Image):

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_label = CLASS_NAMES[
        predicted_class.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    return {
        "disease": predicted_label,
        "confidence": round(
            confidence_value,
            2
        )
    }