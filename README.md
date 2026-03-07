# High-Performance Image Classification with CNN Architectures

## Overview

Built a **scalable image classification pipeline** to evaluate deep learning architectures on a large labeled image dataset.

The project benchmarks 3 approaches:

* **Transfer Learning (Pretrained CNN)**
* **Custom Deep CNN**
* **Fine-Tuned VGG Network**

Focus: **model performance, training efficiency, and reproducible experimentation.**

---

## Results

* Implemented **three CNN architectures** within a unified training pipeline
* Processed **large-scale image dataset with GPU-accelerated training**
* Benchmarked models on **accuracy, convergence speed, and compute cost**
* Built **modular experiment pipeline** enabling controlled model comparison

---

## Architecture Comparison

### 1. Transfer Learning

Pretrained convolutional backbone with a custom classification head.

### 2. Custom Deep CNN

CNN designed and trained from scratch.

* Evaluates performance vs transfer learning
* Understands architecture design trade-offs

### 3. VGG-Based Network

Fine-tuned deep **VGG architecture** for higher representational capacity.

Trade-off:

* Higher accuracy
* Increased compute requirements

---

## Training Pipeline

**Data Pipeline**

* Image normalization
* Data augmentation
* Efficient batch loading

**Training**

* GPU acceleration
* Hyperparameter tuning
* Early stopping

**Evaluation**

* Accuracy
* Precision / Recall
* Training time
* Convergence behavior

---

## Tech Stack

* Python
* PyTorch / TensorFlow
* NumPy
* Matplotlib

## Possible Extensions

* Distributed training
* Automated hyperparameter search
* Deployment for real-time inference
* Evaluation on larger datasets or modern architectures (ResNet, EfficientNet)

```
Model          Accuracy    Training Time
----------------------------------------
Transfer CNN     91%        25 min
Custom CNN       88%        42 min
VGG Fine-tuned   93%        55 min
```
