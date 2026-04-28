import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
from io import BytesIO
import tempfile

# -----------------------------
# Page Style
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UPDATED SIDEBAR (CLEAN)
# -----------------------------
with st.sidebar:

    st.markdown("## 🧠 Brain Tumor AI")

    st.markdown("""
    <div style="background-color:#111827;padding:12px;border-radius:8px">
        <h4 style="color:#22c55e;">📊 Tumor Types</h4>
        <ul>
            <li>Glioma</li>
            <li>Meningioma</li>
            <li>Pituitary</li>
            <li>No Tumor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="background-color:#1f2937;padding:12px;border-radius:8px">
        <h4 style="color:#00d4ff;">ℹ About This App</h4>
        <p>This application uses AI to analyze MRI images and detect brain tumors.</p>
        <p>It provides prediction, confidence score, symptoms, and precautions.</p>
        <p>Reports can also be downloaded in PDF format.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.warning("⚠ For educational use only. Not a medical diagnosis.")
# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_resnet50.keras")
    model.load_weights("resnet_weights.weights.h5")
    return model

model = load_model()

# -----------------------------
# Tumor Details
# -----------------------------
tumor_data = {
    "glioma": {
        "description": "Glioma is a brain tumor that grows aggressively in brain tissue.",
        "symptoms": "Headache, nausea, seizures, memory loss.",
        "precautions": "Regular MRI scans, consult neurologist, avoid stress."
    },
    "meningioma": {
        "description": "Meningioma is usually a non-cancerous tumor.",
        "symptoms": "Vision problems, headache, hearing loss.",
        "precautions": "Routine checkups, surgery if required."
    },
    "pituitary": {
        "description": "Pituitary tumor affects hormone-producing gland.",
        "symptoms": "Hormonal imbalance, fatigue, vision issues.",
        "precautions": "Hormone therapy, medical supervision."
    },
    "no tumor": {
        "description": "No tumor detected.",
        "symptoms": "No abnormal symptoms.",
        "precautions": "Maintain healthy lifestyle."
    }
}

class_names = list(tumor_data.keys())

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(image, prediction, confidence, details):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, "Brain Tumor Detection Report", ln=True, align='C')

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "AI Generated Medical Report", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Description: {details['description']}")
    pdf.multi_cell(0, 10, f"Symptoms: {details['symptoms']}")
    pdf.multi_cell(0, 10, f"Precautions: {details['precautions']}")

    pdf.ln(5)
    pdf.multi_cell(0, 10, "NOTE: This report is AI-generated and not a medical diagnosis.")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp.name)

    pdf.image(temp.name, x=50, w=100)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Brain Tumor Detection System")
st.markdown("### AI-powered MRI analysis for tumor detection")

st.markdown("---")

col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", width="stretch")

# RIGHT SIDE
with col2:
    if uploaded_file:
        with st.spinner("🔍 Analyzing MRI Image..."):
            img = image.resize((128,128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))

        details = tumor_data[predicted_class]

        st.markdown("### 🧾 Diagnosis Result")

        if predicted_class == "no tumor":
            st.success("🟢 Healthy Brain")
        else:
            st.error(f"🔴 Tumor Detected: {predicted_class.upper()}")

        st.write(f"### 📊 Confidence: {confidence:.2f}")
        st.progress(confidence)

        st.warning("⚠ This is an AI prediction. Consult a doctor.")

        st.markdown("---")

        # Probabilities
        st.subheader("📊 Prediction Probabilities")
        for i, cls in enumerate(class_names):
            prob = float(prediction[0][i])
            st.write(f"{cls.capitalize()}: {prob:.2f}")
            st.progress(prob)

        st.markdown("---")

        # Details
        st.subheader("📋 Description")
        st.info(details["description"])

        st.subheader("⚠ Symptoms")
        st.warning(details["symptoms"])

        st.subheader("🛡 Precautions")
        st.success(details["precautions"])

        st.markdown("---")

        # PDF Download
        pdf_file = generate_pdf(image, predicted_class, confidence, details)

        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_file,
            file_name="Brain_Tumor_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please upload an MRI image")