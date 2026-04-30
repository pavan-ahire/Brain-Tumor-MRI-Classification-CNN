import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
from io import BytesIO
import tempfile
import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Brain Tumor AI", layout="wide")

# -----------------------------
# CUSTOM CSS (MODERN UI)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🧠 Brain Tumor AI")

    st.markdown("### 📊 Tumor Types")
    st.info("""
    • Glioma  
    • Meningioma  
    • Pituitary  
    • No Tumor  
    """)

    st.markdown("### 🚀 Features")
    st.success("""
    ✔ AI Detection  
    ✔ Confidence Score  
    ✔ Clinical Insights  
    ✔ PDF Report  
    """)

    st.warning("⚠ Educational Use Only")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_resnet50.keras")
    model.load_weights("resnet_weights.weights.h5")
    return model

model = load_model()

# -----------------------------
# DETAILED MEDICAL DATA
# -----------------------------
tumor_data = {
    "glioma": {
        "description": "Gliomas are malignant tumors originating from glial cells. They are aggressive and tend to infiltrate surrounding brain tissue.",
        "symptoms": "Persistent headaches, seizures, nausea, vomiting, blurred vision, personality changes.",
        "precautions": "Immediate neurologist consultation, MRI monitoring, surgery, chemotherapy or radiation therapy."
    },
    "meningioma": {
        "description": "Meningiomas arise from the meninges and are usually benign, but can cause pressure on brain structures.",
        "symptoms": "Headaches, vision problems, hearing loss, memory issues.",
        "precautions": "Regular monitoring, surgical removal if symptoms worsen."
    },
    "pituitary": {
        "description": "Pituitary tumors affect hormone production and may disrupt endocrine functions.",
        "symptoms": "Hormonal imbalance, fatigue, vision issues, weight changes.",
        "precautions": "Endocrinologist consultation, hormone therapy, MRI follow-ups."
    },
    "no tumor": {
        "description": "No tumor detected. Brain appears structurally normal.",
        "symptoms": "No tumor-related symptoms.",
        "precautions": "Maintain healthy lifestyle and periodic checkups."
    }
}

class_names = list(tumor_data.keys())

# -----------------------------
# PDF GENERATOR (PROFESSIONAL)
# -----------------------------
def generate_pdf(image, prediction, confidence, details):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Brain MRI Diagnostic Report", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "AI-Assisted Clinical Analysis", ln=True, align="C")

    pdf.ln(5)
    pdf.line(10, 30, 200, 30)

    # Info
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Scan Information", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 7, f"Prediction: {prediction.upper()}", ln=True)
    pdf.cell(0, 7, f"Confidence: {confidence:.2f}", ln=True)

    # Risk
    if confidence > 0.75:
        risk = "High"
    elif confidence > 0.4:
        risk = "Moderate"
    else:
        risk = "Low"

    pdf.cell(0, 7, f"Risk Level: {risk}", ln=True)

    # Image (SMALL)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp.name)
    pdf.image(temp.name, x=140, y=40, w=50)

    # Findings
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Clinical Findings", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7,
        f"The MRI scan was analyzed using a deep learning model. "
        f"The system predicts '{prediction.upper()}' with {confidence:.2f} confidence, "
        f"indicating a {risk.lower()} probability of abnormality."
    )

    # Description
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Tumor Description", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, details["description"])

    # Symptoms
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Symptoms", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, details["symptoms"])

    # Precautions
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Precautions", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, details["precautions"])

    # Recommendation
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendation", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7,
        "Consult a neurologist for further evaluation. Additional imaging or tests may be required."
    )

    # Footer
    pdf.ln(5)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6,
        "Disclaimer: This AI-generated report is for educational purposes only."
    )

    return BytesIO(pdf.output(dest='S').encode('latin-1'))

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🧠 Brain Tumor Detection System")
st.markdown("AI-powered MRI Analysis")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Prediction
    img = image.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    details = tumor_data[predicted_class]

    # ---------------- IMAGE ----------------
    st.image(image, caption="MRI Scan", use_container_width=True)

    # ---------------- TOP SUMMARY (NO SCROLL NEEDED) ----------------
    colA, colB, colC = st.columns([1,1,1])

    with colA:
        st.markdown("### Diagnosis")
        if predicted_class == "no tumor":
            st.success("Healthy Brain")
        else:
            st.error(f"{predicted_class.upper()} Detected")

    with colB:
        st.markdown("### Confidence")
        st.write(f"{confidence:.2f}")
        st.progress(confidence)

    with colC:
        st.markdown("### Report")
        pdf = generate_pdf(image, predicted_class, confidence, details)

        st.download_button(
            "⬇ Download Report",
            pdf,
            "Brain_Tumor_Report.pdf"
        )

    st.markdown("---")

    # ---------------- DETAILS BELOW ----------------
    st.markdown("### Prediction Probabilities")
    for i, cls in enumerate(class_names):
        prob = float(prediction[0][i])
        st.write(f"{cls.capitalize()}: {prob:.2f}")
        st.progress(prob)

    st.markdown("---")

    st.markdown("### Description")
    st.info(details["description"])

    st.markdown("### Symptoms")
    st.warning(details["symptoms"])

    st.markdown("### Precautions")
    st.success(details["precautions"])

else:
    st.info("Upload an MRI image to start analysis")