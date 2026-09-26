# 🌊 Edge-AI Autonomous Sonar Debris & Obstacle Detection System
Developed for Smart India Hackathon (SIH)

## 🚀 Project Overview
This project provides an end-to-end, ultra-lightweight computer vision pipeline designed for **Autonomous Underwater Vehicles (AUVs)**. It detects, filters, and geotags underwater debris, discarded objects, and hazards directly on edge hardware using side-scan sonar imagery.

---

## 📂 Repository Structure
```text
sih-sonar-debris/
│
├── data/
│   ├── raw/                 # Raw side-scan sonar image passes
│   └── processed/           # CLAHE-enhanced images, JSON/CSV reports
│
├── model/
│   ├── train_unet.py        # PyTorch U-Net training pipeline
│   ├── export_onnx.py       # Edge deployment model exporter
│   └── sonar_model_edge.onnx# Compiled ONNX model for AUV hardware
│
├── preprocessing/
│   └── preprocess.py        # Contrast enhancement & normalization
│
├── confidence_scoring/
│   └── filter.py            # False-positive suppression & blob scoring
│
├── reporting/
│   └── geotag.py            # Pixel-to-GPS coordinate conversion & reporting
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

---

## 🛠️ Key Technical Features
1. **Preprocessing Pipeline:** Applies Contrast Limited Adaptive Histogram Equalization (**CLAHE**) to make faint sonar targets stand out against noisy water floors.
2. **Lightweight Edge U-Net:** Built with efficiency in mind, optimized via ONNX to run seamlessly on low-power onboard hardware (like Raspberry Pi or NVIDIA Jetson).
3. **False-Positive Suppression:** Filters out background rock formations and sand ripples using aspect ratio heuristics and bounding-box confidence scoring.
4. **Geotagging Engine:** Automatically maps image pixel coordinates to real-world Latitude/Longitude reports (`.json` and `.csv`).

---

## ⚙️ Quick Start Guide
To run this project locally, execute the following commands in your terminal:

- **Install dependencies:**
  pip install -r requirements.txt

- **Generate dummy/test data:**
  python generate_dummy_data.py

- **Run Preprocessing:**
  python preprocessing/preprocess.py

- **Execute Training, Scoring, Geotagging & Export:**
  python model/train_unet.py
  python confidence_scoring/filter.py
  python reporting/geotag.py
  python model/export_onnx.py