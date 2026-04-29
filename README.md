<div align="center">

# 🧠 Brain Tumor MRI Classification using CNN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

> **An AI-powered deep learning system for automated brain tumor detection and classification from MRI scans.**

<img src="https://img.shields.io/github/stars/pavan-ahire/Brain-Tumor-MRI-Classification-CNN?style=social" />
<img src="https://img.shields.io/github/forks/pavan-ahire/Brain-Tumor-MRI-Classification-CNN?style=social" />

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Tumor Categories](#-tumor-categories)
- [Model Architecture](#-model-architecture)
- [How It Works](#️-how-it-works)
- [Technologies Used](#️-technologies-used)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results & Performance](#-results--performance)
- [Features](#-features)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)

---

## 🌟 Overview

Brain tumors are among the most life-threatening medical conditions, and **early detection is critical** to improving patient survival rates. Manual analysis of MRI scans is time-consuming and highly dependent on specialist expertise.

This project presents an **end-to-end deep learning solution** that:
- Automatically analyzes brain MRI images
- Classifies them into 4 distinct categories with high accuracy
- Delivers results instantly through a clean web interface
- Optionally generates a detailed PDF diagnostic report

Built using a **Convolutional Neural Network (CNN)** trained on thousands of labeled MRI images, this system is designed to assist medical professionals and researchers in making faster, more informed decisions.

---

## 🎬 Demo

```
Upload MRI Image → Preprocessing → CNN Inference → Classification Result → (Optional) PDF Report
```

> 💡 **Try it yourself:** Clone the repo, run `streamlit run app.py`, and upload any brain MRI image!

---

## 🧬 Tumor Categories

The model is trained to classify MRI scans into **4 categories**:

| # | Category | Description |
|---|----------|-------------|
| 1 | 🔴 **Glioma Tumor** | Arises from glial cells; most common type of brain tumor, can be aggressive |
| 2 | 🟡 **Meningioma Tumor** | Grows from the meninges (protective brain layers); usually benign but location-dependent |
| 3 | 🔵 **Pituitary Tumor** | Develops in the pituitary gland; affects hormone regulation |
| 4 | 🟢 **No Tumor** | Healthy brain — no abnormality detected |

> **Why these 4?** These represent the most clinically significant and frequently occurring categories in publicly available MRI datasets, making the model practical for real-world screening support.

---

## 🏗️ Model Architecture

The CNN model is built using **TensorFlow/Keras** and follows a sequential architecture:

```
Input Layer (MRI Image)
        ↓
Conv2D → ReLU → MaxPooling
        ↓
Conv2D → ReLU → MaxPooling
        ↓
Conv2D → ReLU → MaxPooling
        ↓
Flatten
        ↓
Dense (Fully Connected) → ReLU → Dropout
        ↓
Dense (Output) → Softmax (4 classes)
```

| Layer Type | Key Purpose |
|---|---|
| **Conv2D** | Extracts spatial features (edges, textures, shapes) |
| **MaxPooling** | Reduces dimensionality; retains dominant features |
| **Dropout** | Prevents overfitting during training |
| **Dense (Softmax)** | Produces probability scores for each tumor class |

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

  [1] User uploads MRI image via Streamlit interface
            ↓
  [2] Image is preprocessed:
        • Resized to model input dimensions
        • Pixel values normalized to [0, 1]
        • Expanded to batch format
            ↓
  [3] Preprocessed image fed to trained CNN model (.keras)
            ↓
  [4] Model outputs probability scores for 4 classes
            ↓
  [5] Class with highest confidence is predicted
            ↓
  [6] Result displayed on screen with confidence percentage
            ↓
  [7] (Optional) PDF report generated with FPDF
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.9+ | Core programming language |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | 2.x | Deep learning framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | Built-in | Model building & training API |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Latest | Interactive web application |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Latest | Numerical operations |
| ![Pillow](https://img.shields.io/badge/-Pillow-3776AB?logo=python&logoColor=white) | Latest | Image loading & preprocessing |
| ![FPDF](https://img.shields.io/badge/-FPDF-4B0082?logoColor=white) | Latest | PDF report generation |

---

## 📂 Project Structure

```
Brain-Tumor-MRI-Classification-CNN/
│
├── 📄 app.py                  # Streamlit web application (UI + inference logic)
├── 🧠 model.keras             # Trained CNN model (saved in Keras format)
├── 📦 requirements.txt        # All Python dependencies
├── 🐍 runtime.txt             # Python version for cloud deployment
└── 📖 README.md               # Project documentation (you are here)
```

> **Note:** The dataset used for training is not included in this repository due to size constraints. Refer to the [Getting Started](#-getting-started) section for data sources.

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have the following installed:
- Python 3.9 or higher
- pip (Python package manager)
- Git

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pavan-ahire/Brain-Tumor-MRI-Classification-CNN.git
cd Brain-Tumor-MRI-Classification-CNN
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

### 5️⃣ Use the App

1. Open your browser at `http://localhost:8501`
2. Upload a brain MRI image (`.jpg`, `.jpeg`, or `.png`)
3. Wait for the model to process and predict
4. View the classification result and confidence score
5. Optionally, download the PDF report

---

## 📊 Results & Performance

> *(Update these values with your actual training results)*

| Metric | Value |
|--------|-------|
| Training Accuracy | ~XX% |
| Validation Accuracy | ~XX% |
| Test Accuracy | ~XX% |
| Loss (Final) | ~X.XX |
| Model Size | ~XX MB |

**Per-class performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | — | — | — |
| Meningioma | — | — | — |
| Pituitary | — | — | — |
| No Tumor | — | — | — |

> 💡 Fill in your actual results from model training logs or `model.evaluate()`.

---

## ✨ Features

- ✅ **Real-time Classification** — Instant predictions on uploaded MRI scans
- ✅ **4-Class Detection** — Glioma, Meningioma, Pituitary, and No Tumor
- ✅ **Confidence Score** — Displays prediction probability for transparency
- ✅ **Interactive Web UI** — Built with Streamlit, no technical knowledge needed
- ✅ **PDF Report Generation** — Downloadable diagnostic summary using FPDF
- ✅ **Image Preprocessing** — Automatic resizing and normalization
- ✅ **Lightweight Model** — Efficient inference, runs on CPU
- ✅ **Deployment Ready** — Includes `runtime.txt` for cloud hosting

---

## 🔮 Future Improvements

- [ ] 🔬 **Grad-CAM Visualization** — Highlight regions of the MRI the model focuses on
- [ ] 📈 **Improved Architecture** — Experiment with ResNet, EfficientNet, or VGG transfer learning
- [ ] 🌐 **Cloud Deployment** — Deploy on Streamlit Cloud, Hugging Face Spaces, or AWS
- [ ] 📊 **Detailed Analytics Dashboard** — Show model confidence charts and history
- [ ] 🗄️ **Multi-format Support** — Accept DICOM (.dcm) format for clinical use
- [ ] 🔄 **Continuous Learning** — Retrain model with new data via feedback loop
- [ ] 📱 **Mobile Optimization** — Responsive UI for mobile browsers

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/YourFeature`
3. **Commit** your changes: `git commit -m 'Add YourFeature'`
4. **Push** to the branch: `git push origin feature/YourFeature`
5. **Open** a Pull Request

Please make sure to update tests and documentation as appropriate.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

> This project is developed **strictly for educational and research purposes**.
>
> It is **NOT** intended to be used as a clinical diagnostic tool or a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified medical professional or radiologist for accurate brain tumor diagnosis.
>
> The predictions made by this model may not be 100% accurate and should never be used to make real medical decisions.

---

## 🙋 Author
**PAVAN AHIRE**
- GitHub: [@your-username](https://github.com/pavan-ahire)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/pavan-ahire-260940364/)
---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

*Made with ❤️ and Python*

</div>
