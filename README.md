# TransFP: Zero-Query Black-Box Adversarial Attacks using Transferability on Object Detection Models

[cite_start]This repository is the official implementation of the paper: **"Zero-Query Black-Box Adversarial Attacks using Transferability on Object Detection Models"**[cite: 1].

## 📌 Overview
[cite_start]This project proposes a novel **zero-query black-box adversarial attack** that aims to induce **false positives** in object detection models (such as YOLOv8 and YOLOv9)[cite: 7, 8, 22]. [cite_start]By creating an ensemble of multiple white-box surrogate models (YOLOv3, v4, v5, v6), we generate a **Universal Adversarial Perturbation (UAP)** with high transferability[cite: 8, 25, 26]. 

[cite_start]Our attack utilizes a combined loss function consisting of **Maximum Object Loss** and **Bounding Box Area Loss** to efficiently bypass Non-Maximum Suppression (NMS) and generate dense false bounding boxes[cite: 4, 28, 29, 124].

<p align="center">
  <img src="docs/overview.png" width="80%" alt="Attack Overview">
</p>

## 🚀 Key Features
- [cite_start]**Zero-Query Black-Box Attack**: No query or feedback is required from the target model[cite: 8, 74].
- [cite_start]**Universal Perturbation (UAP)**: One single perturbation pattern can be applied to various input images in real-time[cite: 70].
- [cite_start]**Dual Optimization**: Combines object confidence boosting and bounding box size reduction to maximize the False Positive (FP) rate[cite: 28, 29, 124].

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
