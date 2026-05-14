# 🛰️ Amazon Deforestation Detection (Sentinel-2 + U-Net)

![Project Status](https://img.shields.io/badge/Status-MVP_Complete-success)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 🌍 The Problem
Illegal deforestation in the Amazon rainforest happens too fast for manual satellite monitoring to catch. By the time human analysts identify a newly bulldozed logging road, the forest is already gone. 

## 💡 The Solution
This project is an automated, end-to-end Deep Learning pipeline that ingests Sentinel-2 satellite imagery and uses a custom **U-Net Semantic Segmentation** model to generate precise, pixel-by-pixel masks of deforested areas in real-time.

---

## 🛠️ Tech Stack
* **Data Engineering:** Google Earth Engine API (GEE)
* **Deep Learning:** TensorFlow / Keras (U-Net Architecture)
* **Computer Vision:** OpenCV, NumPy, PIL
* **Deployment & UI:** Streamlit, Matplotlib

---

## 📂 Architecture & Pipeline

The pipeline is strictly modularized into five core components:

1. **`src/1_download_data.py` (The Harvester):** Interfaces with Google Earth Engine to securely fetch cloud-free (<10%) 8-bit RGB Sentinel-2 imagery and highly accurate Hansen Global Forest Change ground-truth masks for Rondônia, Brazil.
2. **`src/2_split_data.py` (The Organizer):** Strictly partitions the dataset into Training (80%), Validation (10%), and Testing (10%) sets to ensure zero data leakage during evaluation.
3. **`src/3_train_model.py` (The Brain):** Compiles and trains the U-Net architecture. Implements Batch Normalization, Dropout, and dynamic callbacks (`ReduceLROnPlateau`, `EarlyStopping`).
4. **`src/4_predict.py` (The CLI):** A lightweight terminal utility for running rapid inference on random validation images.
5. **`src/app.py` (The Product):** A fully interactive Streamlit dashboard featuring an AI Confidence Heatmap (X-Ray vision) and a dynamic sensitivity threshold slider for edge-case calibration.

---

## 🚧 Engineering Challenges Overcome

Building this system required solving several complex, real-world machine learning roadblocks:

* **The "Broken Data" Crisis:** Raw 16-bit `.tif` satellite files contained corrupted `NaN` pixels that crashed the model. 
  * *Solution:* Re-engineered the ingestion pipeline to process data server-side via the Earth Engine API, delivering clean RGB PNGs.
* **Severe Class Imbalance (The "Shy AI"):** Because >90% of the Amazon is healthy forest, standard loss functions caused the model to predict "100% forest" to artificially inflate accuracy.
  * *Solution:* Implemented a custom **Weighted Binary Cross-Entropy + Dice Loss** function, forcing the model to optimize for mask shape/overlap (IoU) rather than raw pixel counts.
* **Environment Color-Blindness:** Diagnosed a critical matrix misalignment where OpenCV loaded training images in `BGR` format, but the Streamlit UI loaded user inputs in PIL's `RGB` format, destroying model confidence.
  * *Solution:* Engineered a dynamic tensor slicing fix (`image[..., ::-1]`) in the deployment environment to seamlessly translate the color space.

---

## 🚀 How to Run Locally

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt