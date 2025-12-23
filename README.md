# EEG-Powered Brain-Computer Interface for 3D Hand Gesture Control

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Blender](https://img.shields.io/badge/Blender-4.0%2B-orange)
![Emotiv](https://img.shields.io/badge/Hardware-Emotiv%20EPOC%20X-red)
![Status](https://img.shields.io/badge/Status-Capstone%20Complete-green)

## 📌 Project Overview
This project implements an **EEG-based Brain-Computer Interface (BCI)** capable of classifying imagined hand gestures and translating them into **3D model animations** in Blender. 

Utilizing the **Emotiv EPOC X** headset, the system processes 14-channel EEG data to distinguish between four specific motor imagery tasks. The classified signals drive a rigged 3D hand model, serving as a proof-of-concept for neuroprosthetics and rehabilitation applications.

### 🎥 Demo
*Watch the system classifying **Open**, **Fist (Close)**, **Index**, and **Victory** gestures in real-time simulation:*


https://github.com/user-attachments/assets/c34bf7b1-c1ad-4861-a284-94019a007585


### 🎯 Key Objectives
* **Data Acquisition:** Record high-fidelity EEG data for motor imagery tasks.
* **Signal Processing:** Implement a robust pipeline (Bandpass filtering, Artifact removal).
* **Classification:** Train machine learning models (KNN, SVM, RF, MLP) to recognize gestures.
* **Simulation:** Integrate predictions with **Blender** for visual feedback.

---

## 🖐️ Gestures & Dataset
The system classifies **four distinct motor imagery gestures**:
1.  **Open** (Hand Open)
2.  **Close** (Fist)
3.  **Index** (Index finger extension)
4.  **Victory** (Index and Middle finger extension)

* **Participants:** 9 Healthy volunteers (Age 20-22).
* **Trials:** 36 unique trials per class.
* **Hardware:** Emotiv EPOC X (14 Channels, 128Hz Sampling Rate).

---

## 🏗️ System Architecture

The project follows a sequential pipeline architecture:

1.  **Data Layer:** Raw EEG capture via Emotiv Pro.
2.  **Processing Layer (Python):** * **Preprocessing:** 5th-order Butterworth Bandpass Filter (1-45 Hz), Linear Interpolation.
    * **Feature Extraction:** 196 features per window (Statistical, Entropy, Hjorth Parameters, PSD Power Bands).
    * **Classification:** K-Nearest Neighbors (KNN) was identified as the best performer.
3.  **Visualization Layer (Blender):** A Python script inside Blender polls for predictions and animates the rigged hand model.

![Architecture Diagram](architecture.png)

---

## 📊 Results

We evaluated multiple Traditional ML and Deep Learning models. The **K-Nearest Neighbors (KNN)** classifier achieved the highest performance on the test set.

| Model | Accuracy |
| :--- | :--- |
| **KNN** | **97.63%** |
| Random Forest | 95.63% |
| SVM | 93.98% |
| MLP (Deep Learning) | 96.60% |
| Naive Bayes | 33.67% |

*The MLP model performed well with handcrafted features, while end-to-end CNN/LSTM models struggled due to dataset size constraints.*

---

## 🛠️ Installation & Requirements

### 1. Prerequisites
* **Python 3.8+**
* **Blender 3.x or 4.x**
* **Emotiv Pro** (for initial data export)

### 2. Python Dependencies
```bash
pip install numpy pandas scipy scikit-learn joblib nolds
