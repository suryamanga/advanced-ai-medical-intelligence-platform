import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

# Current file:
# backend/llm/report_generator.py

# parents[0] = backend/llm
# parents[1] = backend
# parents[2] = project root

BACKEND_DIR = Path(__file__).resolve().parents[1]

PROJECT_ROOT = BACKEND_DIR.parent


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    ENV_PATH
)


# ============================================================
# GET GROQ API KEY
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GROQ_API_KEY:

    raise ValueError(

        "GROQ_API_KEY not found.\n"
        "Please add your API key to backend/.env"

    )


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

client = Groq(

    api_key=GROQ_API_KEY

)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

LLM_MODEL = "llama-3.1-8b-instant"


# ============================================================
# GENERATE MEDICAL REPORT
# ============================================================

def generate_medical_report(

    prediction,

    confidence

):


    # --------------------------------------------------------
    # CONVERT CONFIDENCE TO PERCENTAGE
    # --------------------------------------------------------

    confidence_percentage = (

        confidence * 100

    )


    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    prompt = f"""

You are an AI medical report assistant.

Generate a concise AI-assisted medical image analysis report
based ONLY on the machine learning prediction provided below.

Machine Learning Model:
EfficientNet-B0

Predicted Condition:
{prediction}

Model Confidence:
{confidence_percentage:.2f}%

The report must contain the following sections:

1. AI Prediction
2. Model Confidence
3. AI-Assisted Interpretation
4. Recommendation

Important instructions:

- Do not claim to be a doctor.
- Do not provide a definitive medical diagnosis.
- Do not invent specific radiological findings.
- Do not mention symptoms that were not provided.
- Clearly state that the result is AI-generated.
- Recommend consultation with a qualified medical professional.
- Keep the report concise and professional.

Generate the report now.

"""


    # ========================================================
    # CALL GROQ LLM
    # ========================================================

    response = client.chat.completions.create(

        model=LLM_MODEL,

        messages=[

            {

                "role": "system",

                "content": (

                    "You are an AI assistant that generates "
                    "safe, concise, AI-assisted medical reports "
                    "from machine learning predictions."

                )

            },

            {

                "role": "user",

                "content": prompt

            }

        ],

        temperature=0.2,

        max_tokens=500

    )


    # ========================================================
    # GET GENERATED REPORT
    # ========================================================

    report = response.choices[

        0

    ].message.content


    return report


# ============================================================
# SAVE REPORT TO FILE
# ============================================================

def save_report(

    report,

    prediction

):


    # --------------------------------------------------------
    # CREATE REPORT DIRECTORY
    # --------------------------------------------------------

    REPORT_DIR = (

        BACKEND_DIR

        / "llm"

        / "reports"

    )


    REPORT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # CREATE FILE NAME
    # --------------------------------------------------------

    file_name = (

        f"{prediction.lower()}_"

        "medical_report.txt"

    )


    report_path = (

        REPORT_DIR

        / file_name

    )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:


        file.write(

            report

        )


    return report_path


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":


    print(

        "\n========================================"

    )


    print(

        "       AI MEDICAL REPORT GENERATOR"

    )


    print(

        "========================================"

    )


    # --------------------------------------------------------
    # TEST VALUES
    #
    # These values come from your EfficientNet-B0 result.
    # --------------------------------------------------------

    prediction = "PNEUMONIA"

    confidence = 0.8270


    print(

        "\nPrediction:",

        prediction

    )


    print(

        "Confidence:",

        f"{confidence * 100:.2f}%"

    )


    print(

        "\nGenerating AI-assisted report..."

    )


    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    report = generate_medical_report(

        prediction,

        confidence

    )


    # --------------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------------

    print(

        "\n========================================"

    )


    print(

        "       AI-ASSISTED MEDICAL REPORT"

    )


    print(

        "========================================"

    )


    print(

        "\n"

        + report

    )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report_path = save_report(

        report,

        prediction

    )


    print(

        "\n========================================"

    )


    print(

        "Report saved at:"

    )


    print(

        report_path

    )


    print(

        "========================================"

    )