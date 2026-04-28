
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
# Class Labels (EDIT if needed)
# -----------------------------
class_names = ["glioma", "meningioma", "no tumor", "pituitary"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Brain Tumor Detection App")
st.write("Upload MRI image to predict tumor type")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    img = image.resize((128, 128))   # ResNet input size
    img_array = np.array(img)

    if img_array.shape[-1] == 4:  # Remove alpha channel if present
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Result
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")