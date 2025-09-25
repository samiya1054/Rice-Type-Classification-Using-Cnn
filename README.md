# Rice Type Classification Using CNN  

## Overview  
The **Rice Type Classification Using CNN** project is an AI-powered tool designed to classify **five rice grain varieties**:  
- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

This project uses **Convolutional Neural Networks (CNN)** with **Transfer Learning (MobileNetV2/4)** to achieve high accuracy, and integrates with a **Flask web application** for an intuitive interface. Users can simply upload an image of rice grains and receive instant predictions.  

---

## Table of Contents
1. [Overview](#-overview)  
2. [System Architecture](#-system-architecture)  
3. [Tech Stack](#-tech-stack)  
4. [Dataset](#-dataset)  
5. [Features](#-features)  
6. [Installation](#-installation)  
7. [Usage](#-usage)   
8. [Advantages & Limitations](#-advantages--limitations)  
9. [Future Scope](#-future-scope)  

---

## System Architecture  
1. **Data Collection & Preprocessing** → Kaggle dataset of rice grains, preprocessing with resizing, normalization, augmentation.  
2. **Model Training** → Transfer learning using **MobileNetV2/4**.  
3. **Evaluation** → Accuracy, confusion matrix, precision/recall, F1-score.  
4. **Deployment** → Flask web app for user interaction.  

---

## Tech Stack  
- **Programming Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **Web Framework:** Flask  
- **Dataset Source:** Kaggle  
- **Others:** NumPy, Matplotlib  

---

## Dataset  
- Source: (https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  
- Classes: Arborio, Basmati, Ipsala, Jasmine, Karacadag  
- Preprocessing:  
  - Resizing → (224 × 224)  
  - Normalization → Scale pixel values [0,1]  
  - Data Augmentation → Rotation, shift, zoom, flip  
  - Noise Reduction → Non-local means denoising  

---

## Features  
- Upload rice grain images for instant classification.  
- Achieves **high accuracy** with MobileNet transfer learning.  
- Flask-powered **user-friendly web interface**.  
- Preprocessing for improved model generalization.  

---

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Rice-Type-Classification-Using-CNN.git
   cd Rice-Type-Classification-Using-CNN


2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:

   ```bash
   python app.py
   ```
5. Open browser at `http://127.0.0.1:5000`

---

## Usage

1. Upload an image of rice grains.
2. The trained model will process the image.
3. Prediction of rice type will be displayed instantly.

---

## Advantages & Limitations

### Advantages:

* High accuracy with transfer learning.
* Cost-effective alternative to expert analysis.
* Scalable for other grain classification.

### Limitations:

* Performance depends on dataset quality.
* Computational resources required for training.
* Deep learning interpretability challenges.

---

## Future Scope

* Real-time classification in rice processing plants.
* Integration with IoT devices for quality monitoring.
* Use of GANs for advanced data augmentation.
* Explainable AI for better interpretability.

---
