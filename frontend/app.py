import streamlit as st
import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# BACKEND_URL = "http://127.0.0.1:8000"

BACKEND_URL = "https://advanced-ai-medical-intelligence-6hij.onrender.com"

# IMPORTANT:
# Change this to the exact endpoint shown in Swagger /docs
# if your endpoint is different.
PREDICT_ENDPOINT = "/predict"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Medical Intelligence Platform",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏥 Advanced AI Medical Intelligence Platform"
)

st.write(
    "Upload a chest X-ray image for AI-powered analysis."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IF IMAGE UPLOADED
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.subheader(
        "📷 Uploaded Medical Image"
    )

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        width=500
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        with st.spinner(
            "Analyzing medical image..."
        ):

            try:

                # ------------------------------------------------
                # PREPARE FILE
                # ------------------------------------------------

                uploaded_file.seek(0)


                files = {

                    "file": (

                        uploaded_file.name,

                        uploaded_file.getvalue(),

                        uploaded_file.type

                    )

                }


                # ------------------------------------------------
                # CALL BACKEND
                # ------------------------------------------------

                response = requests.post(

                    f"{BACKEND_URL}{PREDICT_ENDPOINT}",

                    files=files,

                    timeout=300

                )


                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:


                    result = response.json()


                    # =================================================
                    # GET VALUES FROM BACKEND
                    # =================================================

                    prediction = result.get(

                        "prediction",

                        "UNKNOWN"

                    )


                    confidence = result.get(

                        "confidence",

                        0

                    )


                    gradcam_status = result.get(

                        "gradcam",

                        "Not Generated"

                    )


                    medical_report = result.get(

                        "medical_report",

                        "No medical report available."

                    )


                    heatmap_path = result.get(

                        "heatmap_path",

                        None

                    )


                    overlay_path = result.get(

                        "overlay_path",

                        None

                    )


                    # =================================================
                    # SUCCESS MESSAGE
                    # =================================================

                    st.success(

                        "✅ Analysis completed successfully!"

                    )


                    # =================================================
                    # PREDICTION
                    # =================================================

                    st.header(

                        "🩺 AI Prediction"

                    )


                    col1, col2, col3 = st.columns(

                        3

                    )


                    with col1:

                        st.metric(

                            "Prediction",

                            prediction

                        )


                    with col2:

                        st.metric(

                            "Confidence",

                            f"{float(confidence):.2f}%"

                        )


                    with col3:

                        st.metric(

                            "Grad-CAM",

                            gradcam_status

                        )


                    # =================================================
                    # GRAD-CAM
                    # =================================================

                    st.header(

                        "🔥 Explainable AI - Grad-CAM"

                    )


                    if gradcam_status == "Generated":

                        col1, col2 = st.columns(

                            2

                        )


                        with col1:

                            st.subheader(

                                "Grad-CAM Heatmap"

                            )


                            if heatmap_path:

                                st.info(

                                    "Heatmap generated successfully."

                                )

                                st.code(

                                    heatmap_path,

                                    language="text"

                                )

                                st.caption(

                                    "The backend generated the heatmap. "
                                    "The image will be displayed after "
                                    "the backend exposes the output folder."

                                )

                            else:

                                st.warning(

                                    "Heatmap path not available."

                                )


                        with col2:

                            st.subheader(

                                "Grad-CAM Overlay"

                            )


                            if overlay_path:

                                st.info(

                                    "Grad-CAM overlay generated successfully."

                                )

                                st.code(

                                    overlay_path,

                                    language="text"

                                )

                                st.caption(

                                    "The backend generated the overlay. "
                                    "The image will be displayed after "
                                    "the backend exposes the output folder."

                                )

                            else:

                                st.warning(

                                    "Overlay path not available."

                                )


                    else:

                        st.info(

                            "Grad-CAM was not generated."

                        )


                    # =================================================
                    # MEDICAL REPORT
                    # =================================================

                    st.header(

                        "📄 AI-Assisted Medical Report"

                    )


                    st.markdown(

                        medical_report

                    )


                    # =================================================
                    # API RESPONSE
                    # =================================================

                #
                # =================================================
                # ERROR RESPONSE
                # =================================================

                else:

                    st.error(

                        f"Backend Error "
                        f"({response.status_code}): "
                        f"{response.text}"

                    )


            # ========================================================
            # CONNECTION ERROR
            # ========================================================

            except requests.exceptions.ConnectionError:

                st.error(

                    "❌ Could not connect to FastAPI backend.\n\n"
                    "Please make sure the FastAPI server is running."

                )


            # ========================================================
            # TIMEOUT
            # ========================================================

            except requests.exceptions.Timeout:

                st.error(

                    "⏳ Request timed out. "
                    "Please try again."

                )


            # ========================================================
            # OTHER ERROR
            # ========================================================

            except Exception as e:

                st.error(

                    f"❌ Unexpected error: {str(e)}"

                )