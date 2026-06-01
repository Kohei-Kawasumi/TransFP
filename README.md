# Zero-Query Black-Box Adversarial Attacks using Transferability on Object Detection Models

This repository is the official implementation of the paper: **"Zero-Query Black-Box Adversarial Attacks using Transferability on Object Detection Models"**.

## 📌 Overview
This project proposes a novel **zero-query black-box adversarial attack** that aims to induce **false positives** in object detection models (such as YOLOv8 and YOLOv9). By creating an ensemble of multiple white-box surrogate models (YOLOv3, v4, v5, v6), we generate a **Universal Adversarial Perturbation (UAP)** with high transferability. 

Our attack utilizes a combined loss function consisting of **Maximum Object Loss** and **Bounding Box Area Loss** to efficiently bypass Non-Maximum Suppression (NMS) and generate dense false bounding boxes.


## 🚀 Key Features
- **Zero-Query Black-Box Attack**: No query or feedback is required from the target model.
- **Universal Perturbation (UAP)**: One single perturbation pattern can be applied to various input images in real-time.
- **Dual Optimization**: Combines object confidence boosting and bounding box size reduction to maximize the False Positive (FP) rate.

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ 
- CUDA-enabled GPU

### Clone the Repository
```bash
git clone [https://github.com/Kohei-Kawasumi/TransFP.git](https://github.com/Kohei-Kawasumi/TransFP.git)
cd TransFP
