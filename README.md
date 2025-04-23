# 🧠 Micro-Gesture Classifier

Classifying **micro-gestures** from image sequences using a **hybrid CNN–ResNet50 architecture**. Built to handle imbalanced gesture classes in a real-world dataset of **tennis player interviews**.

> 🔍 Developed for the *Machine vision and Digital Image Analysis* course @ LUT University  
> 📃 Based on the [iMiGUE paper](https://github.com/linuxsino/iMiGUE)  
> 👨‍💻 Contributors: Omer AHMED, Mihály FREI

<p align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python&logoColor=green">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge&logo=pytorch&logoColor=red">
  <img src="https://img.shields.io/badge/Backbone-ResNet50-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Cluster-CSC%20MAHTI-lightgrey?style=for-the-badge">
</p>

---

## 🖼️ Project Overview

This project tackles **micro-gesture classification** using a dataset of 32 gesture classes from **tennis player interviews**.

- Data: variable-length sequences of images per gesture
- Model: **custom CNN + pretrained ResNet50**
- Challenge: **imbalanced dataset**, subtle inter-class differences
- Solution: top-k metrics, early stopping, hybrid model

---

## 🧪 Dataset

- 32 gesture classes  
- Extracted from tennis interviews  
- Sequences of image frames (variable length)  
- Highly **imbalanced** → harder to learn rare gestures  
- Training set provided (no access to test set)
- ⚠️ Dataset not included due to lincensing

---

## ⚙️ Preprocessing & Loading

- All images resized to **224×224**
- Normalized with ImageNet mean/std:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`
- Converted to PyTorch tensors
- Efficient **data loader** handles GPU memory constraints (5GB limit)

---

## 🧠 Model Architecture

> Implemented in **PyTorch**

- **ResNet50** backbone with pretrained weights
- Custom CNN head with:
  - `BatchNorm`, `ReLU`, `AdaptiveAvgPool2d`
  - Final `Linear` layer → 32-class output
- **Early stopping** after 3 non-improving epochs
- Weights saved at each improved validation step

---

## 📊 Evaluation

- **Top-1 Accuracy**: model's best guess  
- **Top-5 Accuracy**: correct label in top 5 predictions  
- Used to better reflect performance on **imbalanced classes**

---

## 📈 Results

- Best weights saved after **epoch 8**
- **Training + Validation** loss/accuracy steadily improved (over 70%)  
- **Top-5 accuracy** remained high despite class imbalance (over 97%)  
- Validation loss slightly >1 due to class imbalance & task complexity

---

## 💡 Future Work

- Handle class imbalance with:
  - More data
  - Data augmentation for sequences
- Try **focal loss** or class-weighted loss
- Explore model **ensembles** (e.g., CNN + skeleton-based)
- Evaluate on the **hidden test set**

---

## 📁 Project Structure

📦 micro-gesture-classifier/  
├── mgc_model.py          
├── test_model.py         
└── requirements.txt           
