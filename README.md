# CV Segmentation Backbones

This repository contains implementations, experiments, and benchmarking of popular image segmentation architectures using PyTorch, Detectron2, and Ultralytics YOLO. The goal is to gain a deeper understanding of segmentation models, their mathematical underpinnings, and performance tradeoffs across semantic, instance, and panoptic segmentation.

---

## Repository Contents

* **down-sampling/** – experiments with pooling and strided convolutions
* **transposed-convolution/** – upsampling blocks for decoder design
* **segmentation-loss-function/** – Dice, BCE, Cross-Entropy, IoU losses
* **custom-unet-training.ipynb** – full U-Net pipeline implementation and training
* **fully\_convolutional\_network.ipynb** – implementation of FCN-8 with VGG16 backbone
* **training-yolov11-instance-segmentation.ipynb** – YOLOv11 instance segmentation training
* **mask-rcnn.ipynb** – pretrained Mask R-CNN with ResNet50-FPN backbone (PyTorch)
* (Planned) **yolov11-instance/** – further instance segmentation experiments

---

## Tech Stack

* Python 3
* PyTorch (core DL framework)
* Detectron2 (for FCN and panoptic segmentation)
* Ultralytics YOLOv11 (instance segmentation)
* Torchvision (for pretrained models like Mask R-CNN)
* NumPy, Pandas, Matplotlib, PIL (data handling & visualization)

---

## Purpose

This repository is educational and experimental in nature:

* Explore segmentation backbones from scratch.
* Compare frameworks (PyTorch vs Detectron2 vs Ultralytics).
* Record quantitative metrics and runtime profiling.
* Build intuition around model design choices (encoder-decoder, skip connections, upsampling).

---

## Evaluation Metrics

We use standard segmentation metrics:

* Dice Coefficient (Dice Loss)
* Binary Cross-Entropy (BCE) Loss
* Mean Intersection-over-Union (mIoU)
* Box Precision, Recall, mAP50, mAP50-95 (for instance segmentation)

---

## Implemented Architectures

### 1. U-Net (Custom PyTorch Implementation)

* Encoder-decoder with skip connections
* Convolutional blocks with Conv + ReLU
* Upsampling via transposed convolutions

**Training Dataset:** Human Segmentation Dataset (Kaggle, GitHub)

**Results:**

* Device: CUDA GPU
* Epochs: 10
* Batch size: 2
* Optimizer: SGD, learning rate = 1e-4

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
* Total: 0.1284s

**Example Prediction:**

* Input: RGB image 512x512
* Output: Binary mask (threshold = 0.4)
* Saved as `output.jpg`

---

### 2. FCN (Fully Convolutional Network)

* Implemented with PyTorch
* Based on VGG16 backbone
* Produces dense per-pixel predictions with skip connections from intermediate layers

**Results:**

* Dummy input: (1, 3, 224, 224)
* Output: (1, 1000, 224, 224)

---

### 3. YOLOv11 (Instance Segmentation)

* Using Ultralytics YOLOv11 framework
* Dataset: COCO subset / custom masks
* Training notebook: `training-yolov11-instance-segmentation.ipynb`
* Features automatic optimizer tuning, AMP, augmentations, caching, plotting, early stopping (patience=5)
* Input image size: 640x640
* Epochs: 100
* Batch size: 8

**Training Snapshot:**

* Box Loss, Segmentation Loss, Classification Loss monitored per epoch
* mAP50, mAP50-95 for both box and mask evaluated on validation set
* Example: Epoch 25 — Box(P:0.766, R:0.715, mAP50:0.72), Mask(P:0.485, R:0.665, mAP50:0.614)

**Results saved in:** `/kaggle/working/YOLOv11-instance-segmentation/train`

---

### 4. Mask R-CNN (Instance Segmentation, PyTorch)

* Implemented with **torchvision.models.detection** `maskrcnn_resnet50_fpn`
* Pretrained on COCO dataset
* Performs instance segmentation with bounding boxes, labels, and masks

**Pipeline:**

* Load pretrained Mask R-CNN with ResNet50-FPN backbone
* Apply preprocessing (ToTensor, resizing)
* Forward pass on input image (e.g., `people_walking.jpg`)
* Extract predictions: bounding boxes, class labels, masks, and scores
* Threshold predictions (default: 0.6)
* Overlay bounding boxes and masks on original image with OpenCV + Matplotlib visualization

**Example Result:**

* Input: People walking street scene
* Output: Instance masks overlaid on persons with bounding boxes and labels

---

## Roadmap

* [x] U-Net implementation & benchmarking
* [x] FCN implementation
* [x] YOLOv11 instance segmentation
* [x] Mask R-CNN instance segmentation (PyTorch)
* [ ] Panoptic segmentation with Detectron2
* [ ] Comparative results table

---

## License

This project is licensed under the MIT License.

---

## Author

Authored by Bhupen. Repository for experimentation, benchmarking, and structured learning in computer vision segmentation.

* [LinkedIn](https://www.linkedin.com/in/bhupenparmar/)
* [GitHub](https://github.com/bhupencoD3)
