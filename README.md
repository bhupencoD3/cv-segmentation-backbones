# CV Segmentation Backbones

This repository contains implementations, experiments, and benchmarking of popular **image segmentation architectures** using **PyTorch**, **Detectron2**, and **Ultralytics YOLO**. The goal is to gain a deeper understanding of segmentation models, their mathematical underpinnings, and performance tradeoffs across semantic, instance, and panoptic segmentation.

---

## 📂 Repository Contents

* **down-sampling/** – experiments with pooling and strided convolutions
* **transposed-convolution/** – upsampling blocks for decoder design
* **segmentation-loss-function/** – Dice, BCE, Cross-Entropy, IoU losses
* **custom-unet-training.ipynb** – full U-Net pipeline implementation and training
* (Planned) **fcn/** – Fully Convolutional Network experiments
* (Planned) **yolov11-instance/** – Instance segmentation with YOLOv11

---

## ⚙️ Tech Stack

* **Python 3**
* **PyTorch** (core DL framework)
* **Detectron2** (for FCN and panoptic segmentation)
* **Ultralytics YOLOv11** (instance segmentation)
* **NumPy**, **Matplotlib**, **PIL** (data handling & visualization)

---

## 🎯 Purpose

This repo is **educational and experimental**:

* Explore segmentation backbones from scratch.
* Compare frameworks (PyTorch vs Detectron2 vs Ultralytics).
* Record **quantitative metrics** and **runtime profiling**.
* Build intuition around model design choices (encoder-decoder, skip connections, upsampling).

---

## 📊 Evaluation Metrics

We use standard segmentation metrics:

* **Dice Coefficient** (Dice Loss)

$\text{Dice}(p, y) = \frac{2 |p \cap y|}{|p| + |y|}$

Loss formulation:

$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}$

* **Binary Cross-Entropy (BCE) Loss**

$\mathcal{L}_{BCE} = - \frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \Big]$

* **Mean Intersection-over-Union (mIoU)**

$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c + FN_c}$

Where \$TP\_c, FP\_c, FN\_c\$ are true positives, false positives, and false negatives for class \$c\$.

---

## 🧩 Implemented Architectures

### 1. U-Net (Custom PyTorch Implementation)

* Encoder-decoder with skip connections
* Convolutional blocks: \$3 \times 3\$ Conv + ReLU
* Upsampling via transposed convolutions

**Training Dataset:** Human Segmentation Dataset (Kaggle, GitHub)

**Results:**

* Training device: CUDA GPU
* Epochs: 10
* Batch size: 2
* Optimizer: SGD, \$lr = 10^{-4}\$

**Training Log (Loss):**

```
Epoch [1/10], Loss: 1.0807
Epoch [5/10], Loss: 1.0804
Epoch [10/10], Loss: 1.0784
```

**Inference Timing:**

* Preprocessing: 0.0068s
* Inference: 0.0022s
* Postprocessing: 0.1194s
* **Total: 0.1284s**

**Example Prediction:**

* Input: RGB image \$512 \times 512\$
* Output: Binary mask (threshold = 0.4)
* Saved as `output.jpg`

---

### 2. FCN (Fully Convolutional Network) – *Coming Soon*

* Implemented with **Detectron2**
* Semantic segmentation baseline
* Will benchmark against U-Net

**Planned Metrics:**

* mIoU per class
* GPU inference latency

---

### 3. YOLOv11 (Instance Segmentation) – *Coming Soon*

* Using **Ultralytics YOLOv11** framework
* Target task: **instance segmentation**
* Dataset: COCO subset / custom masks

**Planned Experiments:**

* Compare instance vs semantic performance
* Real-time inference benchmark (ms per frame)

---

## 📈 Roadmap

* [x] U-Net implementation & benchmarking
* [ ] FCN implementation
* [ ] YOLOv11 instance segmentation
* [ ] Panoptic segmentation with Detectron2
* [ ] Comparative results table

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Authored by Bhupen.
Repository for experimentation, benchmarking, and structured learning in computer vision segmentation.

* [LinkedIn](https://www.linkedin.com/in/bhupenparmar/)
* [GitHub](https://github.com/bhupencoD3)
