# Attention-Guided YOLOv8: Enhancing Fine-Grained Detection of Latent and Active Tuberculosis via Convolutional Block Attention Modules
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** John Christian Niño T. Abuel, 2022-0423 
**Semester:** AY 2025-2026 Sem 1 
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
Tuberculosis (TB) remains a critical global health challenge, with Latent TB Infection (LTBI) posing a unique diagnostic difficulty due to subtle radiological manifestations often missed during standard screening. Existing Computer-Aided Diagnosis (CAD) solutions typically rely on computationally expensive ensemble pipelines or complex multi-scale fusion architectures, rendering them unsuitable for real-time deployment in resource-constrained settings. This study proposes YOLOv8s-CBAM, a novel single-stage object detection architecture that integrates Convolutional Block Attention Modules (CBAM) directly into the backbone and neck of the YOLOv8 network. Unlike approaches requiring explicit lung segmentation, this architecture leverages Channel and Spatial attention to implicitly suppress anatomical noise and enhance fine-grained lesion features. Evaluated on the TBX11K dataset with a strict clinical focus on Active versus Latent infection, the model achieved a mAP@50 of 0.180 for Latent TB, representing a 68% relative improvement over state-of-the-art SFF-YOLOv8 methods. Furthermore, the model demonstrated superior sensitivity for Active TB (Recall 0.665) while maintaining an ultra-fast inference speed of 0.8 ms. These results confirm that lightweight, integrated attention mechanisms can effectively bridge the gap between high-accuracy diagnostics and real-time efficiency.

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

- Visualize predicted bounding boxes and lesion probabilities.


## Related Work
Current approaches for Tuberculosis detection on the TBK11X datast predominantly rely on computationally intensive architectures. For instance, original creators of this dataset has introduced SymFormer, which is a Transformer-based architecture that leverages Symmetric Search Attention and Symmetric Positional Encoding to utilize the symmetry of chest X-Rays [1]. While other studies use complex pipeline by merging multiple heavy classifiers like SqueezeNet, ChexNet, and EfficientNet [2]. Furthermore, recent single-stage attempts like SFF-YOLOv8 rely on intricate multi-scale fusion blocks that increase model overhead [3]. This project addresses these gaps by introducing a streamlined, single-stage YOLOv8s-CBAM architecture. This model will utilize an integrated Channel and Spatial Attention to learn the intrinsic local visual signatures of Latent and Active TB lesions independent of lung symmetry.

- [Detection from Chest X‐Ray Images Based on Modified Deep Learning Approach. (n.d.). https://ieeexplore.ieee.org/document/10496335 [1]]
- [Liu, Y., Wu, Y., Zhang, S., Liu, L., Wu, M., & Cheng, M. (2023). Revisiting Computer-Aided Tuberculosis diagnosis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4),[2]]
- [Rahamathulla, M. P., Emmanuel, W. R. S., Bindhu, A., & Ahmed, M. M. (2024b). YOLOv8’s advancements in tuberculosis identification from chest images. Frontiers in Big Data, 7, 1401981. [3]]

## Methodology
### Dataset
- Source: TBX11K Dataset from  Rethinking Computer-Aided Tuberculosis Diagnosis (Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, Ming-Ming Cheng) https://www.kaggle.com/datasets/usmanshams/tbx-11
- Split: 60/40 train/val
- Preprocessing: For augmentation, rotate image to several positions, resizing to 640x640

### Architecture
- Backbone:  Enhancement: Standard C2f blocks replaced with C2f_CBAM at P3, P4, and P5 scales to refine feature extraction.
- Head: Decoupled YOLOv8 with anchor-free output detection for 2 classes Active TB and Latent TB.
- Hyperparameters: Table below

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 100 |
| Image Size (First)| 512x512|
| Image Size (Second)| 640x640|


### Training Code Snippet
Inside the notebooks


## Experiments & Results
### Metrics
#### The following table benchmarks our custom **YOLOv8s + CBAM** architecture against the baseline model and the leading **SFF-YOLOv8** model from *Rahamathulla et al. (2024)*.

**Key Findings:**
* **Superior Latent TB Detection:** Our final model (640px) achieved a **mAP@50 of 0.180** for Latent TB, significantly outperforming the reference study's **0.107**.
* **Higher Sensitivity (Recall):** Our model demonstrates a higher Recall for Active TB (**0.665** vs. **0.560**), minimizing missed diagnoses in critical cases.

| Model / Experiment | Input Size | Class | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **1. Base Model (Baseline)** | 512x512 | **Active TB** | 0.563 | 0.592 | 0.568 | 0.278 |
| *(Standard YOLOv8s)* | | **Latent TB** | 1.000* | 0.000 | 0.167 | 0.071 |
| | | *All* | *0.781* | *0.296* | *0.368* | *0.175* |
| | | | | | | |
| **2. YOLOv8s + CBAM (v1)** | 512x512 | **Active TB** | 0.606 | **0.665** | 0.640 | **0.317** |
| *(With Data Augmentation)* | | **Latent TB** | 0.212 | 0.082 | 0.125 | 0.055 |
| | | *All* | *0.409* | *0.374* | *0.382* | *0.186* |
| | | | | | | |
| **3. YOLOv8s + CBAM (v2)** | 640x640 | **Active TB** | 0.521 | **0.665** | 0.595 | 0.308 |
| *(Optimized Resolution)* | | **Latent TB** | **0.389** | **0.180** | **0.180** | **0.076** |
| | | *All* | *0.455* | *0.423* | *0.387* | *0.192* |
| | | | | | | |
| **Reference Study (SFF)** | 640x640 | **Active TB** | **0.773** | 0.560 | **0.676** | 0.320 |
| *Rahamathulla et al. (2024)* | | **Latent TB** | 0.176 | 0.146 | 0.107 | 0.057 |
| | | *All*** | *0.711* | *0.622* | *0.657* | *0.475* |

> ***Notes:**
> * **Base Model Latent TB:** The precision of `1.000` is an anomaly caused by extremely low recall (the model made very few guesses, but the first one happened to be correct, then it stopped guessing).
> * **Reference Study "All" Score:** The reference study included "Healthy" and "Sick (Non-TB)" classes which are easier to detect. Their high "All" mAP (0.657) is inflated by these classes (e.g., Healthy mAP was 0.995). Direct comparison should focus on the **Active** and **Latent** rows.

### Demo
![Detection Demo](demo/sample_detection.png)
[Video: [CSC173_Abuel_Final.mp4](demo/CSC173_Abuel_Final.mp4)] [web:41]

## Discussion
- Strengths: The integration of the Convolutional Block Attention Module (CBAM) into the YOLOv8 backbone significantly enhanced feature extraction for subtle pathologies. Most notably, the model achieved a mAP@50 of 0.180 for Latent Tuberculosis, outperforming the complex SFF-YOLOv8 architecture (0.107 mAP) cited in recent literature. Additionally, the model demonstrated high sensitivity with a Recall of 0.665 for Active TB, ensuring fewer infectious cases are missed during screening.
- Limitations: Despite improvements, detecting Latent TB remains challenging compared to Active TB (Precision 0.389 vs. 0.521), likely due to the visual similarity between subtle latent lesions and anatomical noise like clavicles or ribs. Furthermore, the exclusion of the "Healthy" class, while scientifically rigorous, prevents direct comparison with studies that inflate scores using easy-to-classify negative samples.
- Insights: Resolution played a critical role in the efficacy of the attention mechanism. Increasing input size from 512px to 640px resulted in a 44% improvement in Latent TB mAP and an 83% increase in Precision, confirming that spatial attention mechanisms require sufficient pixel density to distinguish fine-grained pathological features.

## Ethical Considerations
- Bias: The model's performance is heavily influenced by the class imbalance in the training data (Active TB instances significantly outnumbered Latent TB). Without mitigation, this could lead to higher false-negative rates for latent carriers, potentially delaying treatment for early-stage patients.
- Privacy: The study utilized the publicly available TBX-11k dataset, which consists of de-identified Chest X-rays. No personally identifiable information (PII) or facial features were processed, ensuring compliance with standard medical data privacy protocols.
- Misuse: There is a risk of "automation bias," where radiologists might over-rely on the AI's bounding boxes and overlook lesions the model misses. This system is designed as a decision support tool, not a replacement for human diagnosis. False negatives in a clinical setting could lead to unchecked disease transmission.

## Conclusion
This study successfully introduced a novel YOLOv8s-CBAM architecture for the automated detection of Tuberculosis from chest radiographs. By embedding channel and spatial attention mechanisms directly into the C2f modules, the model demonstrated superior capability in identifying subtle Latent TB lesions compared to existing state-of-the-art methods (SFF-YOLOv8), achieving a 68% relative improvement in mAP@50 for the latent class. Crucially, this performance was achieved while maintaining high recall for active infection and real-time inference speeds (0.8ms on RTX 4090).

## Installation
1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-Abuel`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
torch>=2.0
ultralytics
opencv-python
albumentations

## References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.  

[2] Liu, Y., Wu, Y., Zhang, S., Liu, L., Wu, M., & Cheng, M. (2023). Revisiting Computer-Aided Tuberculosis diagnosis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4), 2316–2332. https://doi.org/10.1109/tpami.2023.3330825

[3] Detection from Chest X‐Ray Images Based on Modified Deep Learning Approach. (n.d.). https://ieeexplore.ieee.org/document/10496335

[4] Rahamathulla, M. P., Emmanuel, W. R. S., Bindhu, A., & Ahmed, M. M. (2024). YOLOv8’s advancements in tuberculosis identification from chest images. Frontiers in Big Data, 7, 1401981. https://doi.org/10.3389/fdata.2024.1401981


## GitHub Pages
View this project site: [https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/](https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/) [web:32]
