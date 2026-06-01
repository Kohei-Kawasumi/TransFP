## Table1：FP rates of our attacks by various ensemble combinations

| Surrogate Model | Dataset | YOLOv8 ($\epsilon=32$) | YOLOv8 ($\epsilon=16$) | YOLOv9 ($\epsilon=32$) | YOLOv9 ($\epsilon=16$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Benign Image** | COCO2017 | 22.0% | 22.0% | 25.95% | 25.95% |
| | BDD100K | 16.8% | 16.8% | 20.43% | 20.43% |
| **YOLO(v3)** | COCO2017 | 23.4% | 23.1% | 28.0% | 27.6% |
| | BDD100K | 36.4% | 20.9% | 21.6% | 21.0% |
| **YOLO(v4)** | COCO2017 | 23.1% | 23.4% | 28.2% | 27.6% |
| | BDD100K | 27.7% | 21.5% | 20.6% | 20.5% |
| **YOLO(v5)** | COCO2017 | 44.1% | 34.7% | 31.7% | 28.8% |
| | BDD100K | 52.0% | 25.8% | 37.1% | 21.2% |
| **YOLO(v6)** | COCO2017 | 59.7% | 36.3% | 46.4% | 29.8% |
| | BDD100K | 56.3% | 50.2% | 63.8% | 44.4% |
| **YOLO(v3,v4)** | COCO2017 | 46.0% | 25.6% | 29.8% | 29.0% |
| | BDD100K | 46.3% | 24.9% | 21.0% | 20.8% |
| **YOLO(v3,v5)** | COCO2017 | 74.8% | 39.3% | 51.8% | 29.2% |
| | BDD100K | 61.7% | 60.1% | 31.0% | 28.2% |
| **YOLO(v3,v6)** | COCO2017 | 78.0% | 35.7% | 65.1% | 33.1% |
| | BDD100K | 71.5% | 66.4% | 45.4% | 27.3% |
| **YOLO(v4,v5)** | COCO2017 | 66.3% | 27.9% | 50.9% | 29.2% |
| | BDD100K | 49.2% | 47.0% | 34.9% | 31.1% |
| **YOLO(v4,v6)** | COCO2017 | 73.8% | 31.2% | 67.0% | 31.7% |
| | BDD100K | 60.5% | 48.4% | 39.4% | 27.3% |
| **YOLO(v5,v6)** | COCO2017 | 76.0% | 67.2% | **87.2%** | **62.5%** |
| | BDD100K | 56.2% | 51.0% | 76.2% | **73.2%** |
| **YOLO(v3,v4,v5)** | COCO2017 | 74.2% | 57.1% | 42.7% | 35.3% |
| | BDD100K | 68.0% | 39.6% | 21.5% | 22.0% |
| **YOLO(v3,v4,v6)** | COCO2017 | 79.4% | 57.8% | 59.8% | 33.9% |
| | BDD100K | 73.1% | 61.2% | 33.7% | 23.7% |
| **YOLO(v3,v5,v6)** | COCO2017 | **84.4%** | **71.3%** | 80.7% | 51.1% |
| | BDD100K | **82.9%** | 73.5% | **79.7%** | 72.7% |
| **YOLO(v4,v5,v6)** | COCO2017 | 80.5% | 65.3% | 83.6% | 53.7% |
| | BDD100K | 62.1% | 55.2% | 73.4% | 70.7% |
| **YOLO(v3,v4,v5,v6)**| COCO2017 | 84.3% | 62.1% | 77.5% | 50.4% |
| | BDD100K | 74.8% | **73.9%** | 74.2% | 68.9% |



## Table2：Comparison of FP rates of our attacks by only single loss function and both loss functions

| Loss Function | Surrogate Model | Dataset | YOLOv8 ($\epsilon=32$) | YOLOv8 ($\epsilon=16$) | YOLOv9 ($\epsilon=32$) | YOLOv9 ($\epsilon=16$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **bounding box area loss only** | YOLO(v3,v4,v5,v6) | COCO2017 | 25.2% | 22.4% | 28.1% | 25.3% |
| | | BDD100K | 52.3% | 33.8% | 33.4% | 24.4% |
| **max object loss only** | YOLO(v3,v4,v5,v6) | COCO2017 | 78.2% | 56.6% | 72.6% | 44.0% |
| | | BDD100K | 63.7% | 59.8% | 64.0% | 58.7% |
| **both loss functions** | YOLO(v3,v4,v5,v6) | COCO2017 | 84.3% | 62.1% | 77.5% | 50.4% |
| | | BDD100K | 74.8% | 73.9% | 74.2% | 68.9% |



## 📄 Algorithm: Adversarial perturbation generation using multiple surrogate models

### **Inputs & Outputs**
* **Require:**
  * $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_m, y_m)\}$: Input dataset ($m$ images and their labels)
  * $\{\mathcal{M}_1, \mathcal{M}_2, \dots, \mathcal{M}_n\}$: Set of $n$ surrogate models
  * $\epsilon$: Magnitude of perturbation in FGSM
  * $E$: Epoch
  * $\mathcal{L}(\mathcal{M}_i(x), y)$: Function to compute the loss from the output predicted by the surrogate model $\mathcal{M}_i$ by inputting $x$ and a label $y$.
* **Ensure:**
  * $x^\text{adv}$: Final generated attack image

---

### **Procedure: Generation of attack images**

* **Initialization:**
  * $x^\text{adv} \gets \text{None}$
* **Loop Process:**
  * **For** $e \gets 1$ **to** $E$ **do**
    * **For each** $(x, y) \in \mathcal{D}$ **do**
      * $L_\text{total} \gets 0$
      * **For** $i \gets 1$ **to** $n$ **do**
        * $L_i \gets \mathcal{L}(\mathcal{M}_i(x), y)$
        * $L_\text{total} \gets L_\text{total} + L_i$
      * **End For**
      * $$L_\text{avg} = \frac{L_\text{total}}{n}$$
      * $$g \gets \nabla_x L_\text{avg}$$
      * $$\delta \gets \epsilon \cdot \text{sign}(g)$$
      * $$x^\text{adv} \gets x + \delta$$
    * **End For**
  * **End For**
* **Return** $x^\text{adv}$
