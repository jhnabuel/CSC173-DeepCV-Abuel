# [Project Title: e.g., Real-Time Object Detection for Waste Sorting]
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** John Christian Niño T. Abuel, 2022-0423 
**Semester:** AY 2025-2026 Sem 1 
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
[150-250 words: Summarize problem (e.g., "Urban waste sorting in Mindanao"), dataset, deep CV method (e.g., YOLOv8 fine-tuned on custom trash images), key results (e.g., 92% mAP), and contributions.][web:25][web:41]

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Tuberculosis (TB) remains a major health concern, especially in low-resource and high-density regions especially in public hospitals in the Philippines. Chest X-ray interpretation requires trained radiologists, and early signs of TB can be subtle or easily missed. Manual reading is slow, subjective, and prone to diagnostic variability. This project aims to develop an automated system that detects TB lesions directly from chest X-ray images using YOLO object detection, enabling faster and more accessible screening.

### Objectives
- Train a YOLO model to detect TB lesions with bounding boxes.

- Improve upon a proposed model's mAP of 0.657. Achieve 70%+ mAP50 performance.

- Build a complete training pipeline including augmentation, training, and validation.

- Deploy a FastAPI-based web app allowing users to upload X-rays or use a webcam for screening.

- Visualize predicted bounding boxes and lesion probabilities.

![Problem Demo](images/problem_example.gif) [web:41]

## Related Work
- [Paper 1: YOLOv8 for real-time detection [1]]
- [Paper 2: Transfer learning on custom datasets [2]]
- [Gap: Your unique approach, e.g., Mindanao-specific waste classes] [web:25]

## Methodology
### Dataset
- Source: [e.g., Custom 5K images + COCO subset]
- Split: 70/15/15 train/val/test
- Preprocessing: Augmentation, resizing to 640x640 [web:41]

### Architecture
![Model Diagram](images/architecture.png)
- Backbone: [e.g., CSPDarknet53]
- Head: [e.g., YOLO detection layers]
- Hyperparameters: Table below

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Epochs | 100 |
| Optimizer | SGD |

### Training Code Snippet
train.py excerpt
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640)


## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | 85% | 0.87 | 0.82 | 12 |
| **Ours (Fine-tuned)** | **92%** | **0.94** | **0.89** | **15** |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)] [web:41]

## Discussion
- Strengths: [e.g., Handles occluded trash well]
- Limitations: [e.g., Low-light performance]
- Insights: [e.g., Data augmentation boosted +7% mAP] [web:25]

## Ethical Considerations
- Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
- Privacy: No faces in training data
- Misuse: Potential for surveillance if repurposed [web:41]

## Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

## Installation
1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-YourLastName`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
torch>=2.0
ultralytics
opencv-python
albumentations

## References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.  
[2] Deng, J., et al. "ImageNet: A large-scale hierarchical image database," CVPR, 2009. [web:25]

## GitHub Pages
View this project site: [https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/](https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/) [web:32]
