
# CSC173 Deep Computer Vision Project Proposal
**Student:** John Christian Niño T. Abuel, 2022-0423 
**Date:** December 10, 2025

## 1. Project Title 
YOLO-Based Tuberculosis Lesion Detection in Chest X-Rays

## 2. Problem Statement
Tuberculosis (TB) remains a major health concern, especially in low-resource and high-density regions especially in public hospitals in the Philippines. Chest X-ray interpretation requires trained radiologists, and early signs of TB can be subtle or easily missed. Manual reading is slow, subjective, and prone to diagnostic variability. This project aims to develop an automated system that detects TB lesions directly from chest X-ray images using YOLO object detection, enabling faster and more accessible screening.


## 3. Objectives
- Train a YOLO model to detect TB lesions with bounding boxes.

- Improve upon a proposed model's mAP of 0.657. Achieve 70%+ mAP50 performance.

- Implement a Convolutional Block Attention Module (CBAM) for attention mechanism augmentation.

- Build a complete training pipeline including augmentation, training, and validation.

- Deploy a FastAPI-based web app allowing users to upload X-rays or use a webcam for screening.

- Visualize predicted bounding boxes and lesion probabilities.

## 4. Dataset Plan
- Source:TBX11K Dataset (https://www.kaggle.com/datasets/omareldash75/tbx11k-original-dataset-with-bounding-boxes/data)
- Classes: 
  - Healthy	
  - Sick & Non-TB	
  - TB/Active TB	
  - Latent TB
  - Active & Latent TB	
  - Uncertain TB
- Acquisition: (https://www.kaggle.com/datasets/omareldash75/tbx11k-original-dataset-with-bounding-boxes/data)

## 5. Technical Approach
- Architecture sketch
- Model: YOLOv8
- Framework: PyTorch
- Hardware: Google Colab

## 6. Expected Challenges & Mitigations
- Challenge: Small dataset for Latent TB
- Solution: Augmentation 
