<div align="center">

#   **SignSense: Real-Time Transformer Vision** 
**A modern, end-to-end Sign Language Recognition Pipeline built on custom Object Detection Transformers (DETR).**

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Albumentations](https://img.shields.io/badge/albumentations-%23FFD000.svg?style=for-the-badge)](https://albumentations.ai/)

<br/>

</div>

---

## 🚀 **Overview**

**SignSense** showcases the power of adapting cutting-edge Transformer architectures to dynamic vision tasks. By implementing a custom **DEtection TRansformer (DETR)**, this repository provides a full-lifecycle walkthrough for interpreting sign language in real-world, real-time webcam feeds. 

From data processing and model definition to robust training, validation, and a sleek kiosk-style inference endpoint, SignSense delivers a zero-noise, high-precision detection experience.

---

## 🧠 **Model Architecture**

SignSense is built on a custom **DETR** pipeline specifically tuned for temporal stability and high-confidence spatial understanding.

*   **Backbone:** `ResNet-50` (ImageNet-pre-trained) for deep feature extraction.
*   **Positional Embedding:** Dynamic 2D Sine-Cosine encodings injected directly into the feature maps to preserve critical spatial coordinate hierarchies.
*   **Transformer Core:** PyTorch's native multi-head attention (`nn.Transformer`) mapping dense image features to discrete object queries.
*   **Prediction Heads:** Fully connected networks resolving transformer latent space into bounding box coordinates (`sigmoid`) and class multi-class probabilities.

---

## ⚙️ **Quick Start**

Get the real-time inference engine up and running in minutes.

### 1. **Clone & Enter Environment**

```bash
git clone https://github.com/your-username/SignSense.git
cd SignSense
```

### 2. **Install Dependencies**

Ensure Python `3.13+` is active. This project utilizes native `pyproject.toml` standards.

```bash
pip install .
```
*(Tip: Consider using a virtual environment like `uv` or `venv`)*

### 3. **Launch the Real-Time Feed**

With your `4426_model.pt` weights placed in `/pretrained`, spark up the webcam feed!

```bash
python src/realtime.py
```
*(Make sure your gestures are clearly visible in frame for optimal recognition).*

---

## 📊 **Evaluation & Metrics**

SignSense is built for rigorous multi-class detection stability. The model has achieved exceptional metric integrity on our validation subsets.

### **Intersection over Union (IoU) Precision**

| Metric | Score | IoU Target | Description |
| :--- | :---: | :---: | :--- |
| **mAP** | `0.7024` | 0.50 : 0.95 | Overall precision across all standard bounding box thresholds. |
| **mAP@50** | **`1.0000`** | 0.50 | Perfect spatial overlap at the standard confidence bound. |
| **mAP@75** | `0.7726` | 0.75 | High stability across strict bounding box criteria. |

### **Classification Integrity**

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Macro F1 Score (Top-1)** | **`1.0000`** | Perfect classification balancing false positives and false negatives on tested states. |

### **Inference Performance**

| Metric | Value | Engine Context |
| :--- | :---: | :--- |
| **Inference Latency** | `~148.13 ms` | Core `model.forward()` cycle time. |
| **Effective FPS** | `~6.75` | Real-world application frame-rate (including GUI renders and smoothers). |

<br/>

---

## 🔧 **Core Scripts**

Dive deeper into the project internals via the `src/` directory:

*   **`src/train.py`** & **`src/train_scratch.py`** — Complete boilerplate to re-train the core model with logging and checkpointing handles.
*   **`src/evaluate_model.py`** — Metric computation pipelines driving the precision scores detailed above.
*   **`src/realtime.py`** — The production-ready looping GUI script powering local webcam inference.

