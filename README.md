# Advanced AI Medical Intelligence Platform

An AI-powered medical image analysis platform for chest X-ray classification. The system uses a pretrained **EfficientNet-B0** deep learning model, **Grad-CAM** for visual explainability, an **LLM-based medical report generator**, **FastAPI** for backend REST APIs, and **Streamlit** for the user interface.

> **Important Medical Disclaimer:** This project is intended for educational and research purposes only. It is not a medical device and must not be used as a substitute for diagnosis, treatment, or professional medical advice. AI predictions should always be reviewed by qualified healthcare professionals.

---

## 📌 Project Overview

The **Advanced AI Medical Intelligence Platform** is designed to analyze chest X-ray images and provide an AI-assisted interpretation workflow.

The platform performs the following major steps:

1. Accepts a chest X-ray image from the user.
2. Preprocesses the image for deep learning inference.
3. Uses EfficientNet-B0 to classify the image as:
   - `NORMAL`
   - `PNEUMONIA`
4. Calculates the model confidence.
5. Generates a Grad-CAM visualization to explain the regions influencing the prediction.
6. Generates an AI-assisted medical report using an LLM.
7. Stores prediction information for prediction history.
8. Exposes the functionality through a FastAPI REST API.
9. Provides a user-friendly Streamlit frontend.

---

## ✨ Key Features

- 🩻 Chest X-ray image classification
- 🧠 EfficientNet-B0 deep learning model
- 🔍 Binary classification: NORMAL vs PNEUMONIA
- 📊 Model evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report
- 🔥 Grad-CAM explainable AI visualization
- 🤖 LLM-assisted medical report generation
- ⚡ FastAPI REST API
- 🖥️ Streamlit frontend
- 🗃️ Prediction history database
- 📄 Medical report generation
- 🔐 Environment variable support using `.env`
- 🧩 Modular project architecture

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      User            │
                         │ Upload Chest X-ray   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Frontend   │
                         └──────────┬───────────┘
                                    │ HTTP Request
                                    ▼
                         ┌──────────────────────┐
                         │ FastAPI Backend      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ EfficientNet-B0               │
                    │ Image Classification         │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                ┌────────────────┐    ┌────────────────┐
                │ Prediction     │    │ Confidence     │
                └───────┬────────┘    └───────┬────────┘
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Grad-CAM / XAI       │
                         │ Visual Explanation   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ LLM Medical Report   │
                         │ Generation           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Database             │
                         │ Prediction History   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Final Results        │
                         │ Prediction + XAI +   │
                         │ Medical Report       │
                         └──────────────────────┘
```

---

## 🧠 Machine Learning Model

### EfficientNet-B0

The project uses **EfficientNet-B0** as the image classification backbone.

The model is initialized with pretrained ImageNet weights and adapted for binary chest X-ray classification.

### Output Classes

```text
0 → NORMAL
1 → PNEUMONIA
```

### Why EfficientNet-B0?

EfficientNet-B0 was selected because it provides a good balance between:

- Model accuracy
- Computational efficiency
- Number of parameters
- Inference speed
- Suitability for transfer learning

The project uses transfer learning by leveraging pretrained visual features and adapting the classifier for the project's two classes.

---

## 📂 Project Structure

The project follows a modular architecture similar to the following:

```text
Advanced AI Medical Intelligence Platform/
│
├── backend/
│   ├── main.py
│   │
│   ├── model/
│   │   ├── model.py
│   │   └── trained_model.pth
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── dataset.py
│   │   └── evaluate.py
│   │
│   ├── xai/
│   │   ├── gradcam.py
│   │   └── outputs/
│   │
│   ├── llm/
│   │   ├── medical_report.py
│   │   └── reports/
│   │
│   └── database/
│       └── ...
│
├── frontend/
│   └── app.py
│
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       │
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       │
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── .env
├── requirements.txt
├── README.md
└── ...
```

> The exact folder structure may vary depending on the final implementation.

---

## 📊 Dataset

The project uses a chest X-ray dataset organized into three subsets:

```text
train/
val/
test/
```

Each subset contains:

```text
NORMAL/
PNEUMONIA/
```

The dataset is used for binary image classification.

### Dataset Organization

```text
Chest X-ray Dataset
│
├── Training Data
│   ├── NORMAL
│   └── PNEUMONIA
│
├── Validation Data
│   ├── NORMAL
│   └── PNEUMONIA
│
└── Testing Data
    ├── NORMAL
    └── PNEUMONIA
```

---

## 🔄 Image Preprocessing

Input images are transformed before being passed to EfficientNet-B0.

The preprocessing pipeline includes:

- RGB conversion
- Image resizing to `224 × 224`
- Conversion to PyTorch tensor
- ImageNet normalization

The normalization values used are:

```text
Mean:
[0.485, 0.456, 0.406]

Standard Deviation:
[0.229, 0.224, 0.225]
```

---

## 🏋️ Model Training

The model is trained using PyTorch.

### Training Configuration

```text
Model: EfficientNet-B0
Number of Classes: 2
Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 0.001
Epochs: 5
Backbone: Frozen during initial classifier training
```

The trained model is saved as:

```text
backend/model/trained_model.pth
```

### Run Training

From the `backend` directory:

```bash
python training/train.py
```

The training process reports:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

The best model based on validation accuracy is saved automatically.

---

## 📈 Model Evaluation

The model is evaluated using the test dataset.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Run:

```bash
python training/evaluate.py
```

### Example Evaluation Result

The current project evaluation produced approximately:

```text
Test Loss: 0.3909
Accuracy: 84.13%
Precision: 82.12%
Recall: 95.38%
F1 Score: 88.26%
```

Confusion Matrix:

```text
                 Predicted
                 NORMAL  PNEUMONIA

Actual NORMAL       153       81
Actual PNEUMONIA     18      372
```

### Interpretation

```text
True Negative (TN): 153
False Positive (FP): 81
False Negative (FN): 18
True Positive (TP): 372
```

For this binary classification task:

- **False Negative:** A pneumonia image is predicted as NORMAL.
- **False Positive:** A NORMAL image is predicted as PNEUMONIA.

> These metrics are specific to the current experiment and may change if the dataset, preprocessing, training configuration, or model weights are changed.

---

## 🔥 Explainable AI with Grad-CAM

The platform uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model prediction.

Grad-CAM highlights image regions that contributed to the model's classification decision.

### Workflow

```text
Input X-ray
     │
     ▼
EfficientNet-B0
     │
     ▼
Predicted Class
     │
     ▼
Target Convolutional Layer
     │
     ▼
Gradient Calculation
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Heatmap Overlay
```

The generated files are stored in:

```text
backend/xai/outputs/
```

Example:

```text
image_heatmap.jpg
image_gradcam.jpg
```

Grad-CAM is intended to improve model interpretability and does not represent a definitive clinical explanation.

---

## 🤖 AI-Assisted Medical Report Generation

After classification, the platform generates an AI-assisted medical report.

The report can contain:

1. AI prediction
2. Model confidence
3. AI-assisted interpretation
4. Recommendation
5. Medical disclaimer

Example workflow:

```text
X-ray Image
     │
     ▼
EfficientNet-B0
     │
     ▼
Prediction + Confidence
     │
     ▼
LLM Report Generator
     │
     ▼
AI-Assisted Medical Report
```

Reports are stored in:

```text
backend/llm/reports/
```

> The generated report is not a medical diagnosis. It is an AI-assisted interpretation and must be reviewed by a qualified healthcare professional.

---

## ⚡ FastAPI Backend

The backend provides REST API endpoints for image analysis and prediction history.

The backend is implemented using FastAPI.

### Start the Backend

From the `backend` directory:

```bash
python -m uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Main API Workflow

```text
POST image
    │
    ▼
FastAPI
    │
    ├── EfficientNet-B0 Prediction
    │
    ├── Confidence Calculation
    │
    ├── Grad-CAM Generation
    │
    ├── LLM Medical Report
    │
    └── Save Prediction History
             │
             ▼
       JSON Response
```

### Example Prediction Response

```json
{
  "success": true,
  "prediction_id": 1,
  "filename": "example.jpeg",
  "prediction": "PNEUMONIA",
  "confidence": 99.96,
  "gradcam": "Generated",
  "heatmap_path": "path/to/heatmap.jpg",
  "overlay_path": "path/to/gradcam.jpg",
  "medical_report": "AI-assisted medical report..."
}
```

### Prediction History

The prediction history endpoint can be used to retrieve previous prediction records.

Example:

```text
GET /predictions
```

---

## 🖥️ Streamlit Frontend

The frontend provides a simple user interface where users can:

- Upload a chest X-ray image
- View the uploaded image
- Send the image to the FastAPI backend
- View the prediction
- View confidence
- View Grad-CAM explanation
- View the AI-assisted medical report

Start the frontend with:

```bash
streamlit run frontend/app.py
```

The Streamlit application normally opens at:

```text
http://localhost:8501
```

---

## 🔗 Frontend and Backend Integration

The frontend communicates with the FastAPI backend using HTTP requests.

```text
Streamlit
    │
    │ POST image
    ▼
FastAPI
    │
    ▼
AI Pipeline
    │
    ├── Prediction
    ├── Confidence
    ├── Grad-CAM
    ├── Medical Report
    └── Database
    │
    ▼
JSON Response
    │
    ▼
Streamlit UI
```

Make sure the FastAPI backend is running before starting image analysis from Streamlit.

---

## 🔐 Environment Variables

If the LLM report generation requires an API key, store the key in a `.env` file.

Example:

```env
GROQ_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
*.pth
```

> Never expose API keys or other secrets in source code or public repositories.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the Project

```bash
cd "Advanced AI Medical Intelligence Platform"
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Complete Project

### Terminal 1 — Start FastAPI

```bash
cd backend
python -m uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2 — Start Streamlit

Open another terminal in the project root:

```bash
streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

---

## 🧪 Complete Application Flow

```text
1. User opens Streamlit frontend
          │
          ▼
2. User uploads chest X-ray
          │
          ▼
3. Frontend sends image to FastAPI
          │
          ▼
4. Backend preprocesses image
          │
          ▼
5. EfficientNet-B0 predicts class
          │
          ▼
6. Confidence score is calculated
          │
          ▼
7. Grad-CAM generates explanation
          │
          ▼
8. LLM generates AI-assisted report
          │
          ▼
9. Prediction is saved in database
          │
          ▼
10. Backend returns results
          │
          ▼
11. Frontend displays:
       - Prediction
       - Confidence
       - Grad-CAM
       - Medical Report
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision
- EfficientNet-B0

### Machine Learning / Data Processing

- NumPy
- Pandas
- Scikit-learn

### Computer Vision

- OpenCV
- PIL / Pillow

### Explainable AI

- Grad-CAM

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### LLM / Generative AI

- LLM API
- Python-based report generation

### Database

- Database layer for prediction history

### Configuration

- python-dotenv

---

## 📋 Current Project Status

| Component | Status |
|---|---|
| Dataset | ✅ Completed |
| EfficientNet-B0 Training | ✅ Completed |
| Model Evaluation | ✅ Completed |
| Grad-CAM / XAI | ✅ Completed |
| LLM Medical Report | ✅ Completed |
| FastAPI REST API | ✅ Completed |
| Prediction History Database | ✅ Completed |
| Frontend Interface | ✅ Completed |
| Frontend + Backend Integration | ✅ Completed |
| README Documentation | ✅ Completed |
| PDF Project Report | ⏳ Pending |
| requirements.txt | ⏳ Pending / Final Verification |
| Dockerfile | ⏳ Optional |
| Final Testing | ⏳ Pending |
| GitHub Repository | ⏳ Pending |
| Deployment | ⏳ Optional |

---

## ⚠️ Limitations

- The model is trained for only two classes: NORMAL and PNEUMONIA.
- The model should not be considered a general-purpose medical diagnostic system.
- Dataset bias may affect model performance.
- Model confidence does not represent clinical certainty.
- Grad-CAM visualizations are explanations of model behavior, not clinical evidence.
- LLM-generated reports may contain errors or incomplete interpretations.
- The system requires professional medical review before any clinical decision.

---

## 🚀 Future Improvements

Possible future enhancements include:

- Multi-disease classification
- More comprehensive medical datasets
- Model fine-tuning and hyperparameter optimization
- Improved class balancing
- Advanced data augmentation
- Better clinical validation
- User authentication
- Patient profile management
- Detailed prediction history dashboard
- Downloadable PDF medical reports
- Docker containerization
- Cloud deployment
- Monitoring and logging
- Model versioning
- MLOps pipeline
- Improved security and privacy controls
- Integration with hospital information systems

---

## 📄 Project Purpose

This project demonstrates the integration of:

```text
Deep Learning
      +
Computer Vision
      +
Explainable AI
      +
Generative AI
      +
REST APIs
      +
Frontend Development
      +
Database
```

The goal is to build an end-to-end AI application that demonstrates how a medical image classification model can be combined with explainability and AI-assisted reporting in a single platform.

---

## 👨‍💻 Author

**Surya Manga**

B.Tech — Electronics and Communication Engineering

---

## 📜 Disclaimer

This software is developed for educational, research, and demonstration purposes.

The predictions, visual explanations, and generated reports are produced by artificial intelligence systems and are not intended to provide medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical decisions.

---

## ⭐ Acknowledgements

This project uses open-source technologies and deep learning frameworks including PyTorch, Torchvision, FastAPI, Streamlit, OpenCV, and other Python libraries.

