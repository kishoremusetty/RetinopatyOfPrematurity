# 🧠 Retinopathy of Prematurity Detection using Hybrid Quantum-Classical AI

This project presents a **hybrid deep learning + quantum computing framework** for detecting **Retinopathy of Prematurity (ROP)** using retinal fundus images.

It combines:

* 🧬 **Segmentation (U-Net + ResNet)**
* 🌐 **Global Feature Extraction (Vision Transformer)**
* ⚛️ **Quantum Artificial Neural Network (QANN)**
* 📊 **Streamlit Web App for Deployment**

---

## 📌 Problem Statement

Retinopathy of Prematurity (ROP) is a serious eye disease in premature infants that can lead to blindness if not detected early.

Traditional diagnosis:

* Time-consuming
* Depends on expert skill
* Prone to human error

👉 This project automates ROP detection using AI + Quantum Learning.

---

## 🧠 Proposed Architecture

The system follows a **4-stage pipeline**:

### 🔹 1. Feature Extraction

* Local Features → U-Net segmentation (ridge, vessels)
* Global Features → Vision Transformer (ViT)

### 🔹 2. Feature Fusion

* Combines local + global features
* Uses attention + compression

### 🔹 3. Quantum Processing

* Variational Quantum Circuit (VQC)
* Hybrid encoding (Angle + Amplitude)

### 🔹 4. Decision Layer

* Sigmoid classifier
* Output → Normal / Abnormal

📌 (Refer to architecture diagram in the repo)

---

## 📂 Project Structure

```
RetinopathyOfPrematurity/
│
├── Copy_of_done.ipynb   # ⚛️ Quantum Model (QANN Implementation)
├── Segmentation.ipynb   # 🧬 Image Segmentation (U-Net + Ridge Extraction)
├── app.py               # 🌐 Streamlit Web Application
└── README.md
```

---

## ⚙️ Modules Explanation

### ⚛️ Quantum Module (`Copy_of_done.ipynb`)

* Implements **Quantum Artificial Neural Network**
* Uses:

  * Angle Encoding
  * Amplitude Encoding
  * Variational Quantum Circuits
* Performs final classification

---

### 🧬 Segmentation Module (`Segmentation.ipynb`)

* Uses **U-Net architecture**
* Extracts:

  * Blood vessels
  * Ridge structures
* Applies **ResNet for feature learning**

---

### 🌐 Web App (`app.py`)

* Built using **Streamlit**
* Allows:

  * Upload retinal image
  * Run prediction
  * Display result (Normal / Abnormal)

---

## 📊 Dataset

* Dataset used: **HVDROPDB**
* Type:

  * Retinal fundus images
  * Segmentation masks
* Classes:

  * Normal
  * Abnormal (ROP)

---

## 📈 Results

From the research paper:

* ✅ Accuracy: **95.8%**
* ✅ F1 Score: **0.957**
* ✅ Sensitivity: **92.86%**
* ✅ Specificity: **96.55%**

output:
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/36e5c211-f889-4dfe-abc1-229b3d0fd4c5" />

---

## 🚀 How to Run

### 1️ Clone Repo

```
git clone https://github.com/your-username/RetinopathyOfPrematurity.git
cd RetinopathyOfPrematurity
```

## 📥 Model Weights (Updated)

Download the required models:

* 🧬 Segmentation Model:
  https://drive.google.com/file/d/1nhP_NzMBj1EMYui5WssUiLhVrn8cJoEw/view?usp=sharing

* ⚛️ Quantum Model:
  https://drive.google.com/file/d/1wiHfvbVPZpgvtetxcpoZ2Xt4jb4aoc70/view?usp=sharing

---

## 📁 Place the Files Here

After downloading, place them in:

```
models/
├── segment.pth
├── quantum.pth
```

---

## ⚙️ Changes in `app.py`

Update model paths:

```python
SEGMENT_MODEL_PATH = "models/segment.pth"
QUANTUM_MODEL_PATH = "models/quantum.pth"
```

---

## 🧠 Load Both Models

```python
segment_model = load_model(SEGMENT_MODEL_PATH)
quantum_model = load_model(QUANTUM_MODEL_PATH)
```

---

## 🔄 Prediction Flow Update

```python
# Step 1: Segmentation
segmented_output = segment_model(image_tensor)

# Step 2: Quantum Prediction
prob = predict(quantum_model, segmented_output)
```

---
## ▶️ Run the Application

Start the Streamlit app using:

```bash
streamlit run app.py
```

After running, open your browser at:

```
http://localhost:8501
```

---

### ⚠️ Before Running

Make sure:

* Models are downloaded and placed in `models/` folder
* All dependencies are installed

---

### 🧪 Usage

1. Upload a retinal image
2. Wait for processing
3. View result:

   * ✅ Normal
   * ⚠️ Abnormal (ROP Detected)

---


## ⚠️ Note

Both models are required for the full pipeline:

* Segmentation → Feature Extraction
* Quantum Model → Final Classification

---


## 🧪 Tech Stack

* Python
* PyTorch / TensorFlow
* OpenCV
* Streamlit
* PennyLane / Qiskit (for Quantum Computing)

---

## 🎯 Key Contributions

✔ Hybrid Quantum-Classical Architecture
✔ Ridge-based ROP Analysis
✔ Efficient Learning on Small Dataset (N=185)
✔ Improved Early Detection Accuracy

---

## 📌 Future Work

* Use larger multi-hospital datasets
* Deploy as cloud-based diagnostic tool
* Improve explainability (Grad-CAM + Quantum insights)

---

## 👨‍💻 Authors

* Krishna Santosh Naidana
* Sohith Sai Sanaka
* Dedeepya Adduri
* Bhargav Bysani
* **Kishore Musetty**

---

## 📜 License

This project is for **research and academic purposes**.

Dataset used under:

* CC BY 4.0 License (HVDROPDB)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share!

---
