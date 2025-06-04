# 🙋‍♀️ About the Author
This project was developed as part of my machine learning portfolio to demonstrate proficiency in:
- CNN model design and evaluation
- ML preprocessing pipelines
- Hands-on deployment using TensorFlow and Gradio

# 📜 License
This project is for educational purposes. Data sources remain the property of their respective owners.

# 🧠 Brain Tumor Detection from MRI Images using CNN

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify brain MRI scans as either **tumorous** or **healthy**. It was developed as a personal machine learning portfolio project using a dataset of **253 images** (155 with brain tumors, 98 without), trained in Google Colab, and deployed via a Gradio interface for interactive testing.

> ✅ Built for showcasing ML model development, data preprocessing, and deployment via Gradio.

## 📌 Project Highlights

- Trained a binary image classifier using CNN on grayscale brain MRI scans
- Achieved 100% test accuracy on the evaluation set (see limitations below)
- Deployed with a user-friendly **Gradio** interface for live image classification
- Built and tested entirely in **Google Colab**

## 🗂️ Dataset Overview

- Total images used: **253**
  - **155** MRI images with brain tumors
  - **98** MRI images of healthy brains
- Images were pre-cleaned and labeled into two classes: `yes` (tumor), `no` (healthy)
- Source: provided by course instructor

## 🔎 Data Preprocessing

- All images were:
  - Resized to **150x150 pixels**
  - Converted to **grayscale**
  - Normalized to scale pixel values between 0 and 1
- Used **ImageDataGenerator** for basic data augmentation (e.g. rescale)

> ⚖️ **No class balancing was applied.**
Although the dataset had more tumor images (155 vs 98), the class imbalance was not severe enough to cause biased learning. Evaluation metrics and model performance showed the classifier handled both classes reliably without oversampling or undersampling.

## 🧠 Model Architecture

Sequential CNN Model:
- Conv2D → MaxPooling2D
- Conv2D → MaxPooling2D
- Flatten
- Dense (128) + ReLU
- Dropout (0.5)
- Dense (1) + Sigmoid

## 🧠 Compile Model
- Loss function: binary_crossentropy
- Optimizer: Adam
- Evaluation metric: accuracy
- Epochs trained: 10

## ✅ Model Performance
- Test accuracy: 1.0000
Test loss: 0.0708

## 📌 Limitations
- Small dataset (253 images) limits generalizability
- Grayscale images only, no contrast enhancements or 3D imaging
- Achieved high test accuracy on a small test set; future work could involve cross-validation, regularization tuning, or external validation datasets

brain-scan-machine-learning/
├── Brain_tumour_detection_using_CNN.ipynb     # Main notebook
├── training images/                           # Optional: test images
    └── no                                     # Healthy brain scans
    └── yes                                    # Tumour brain scans
    └── Brain scan to test trained model       # Image used to test trained model in Gradio App at the end
├── Gradio app screen shot                     # Gradio app screenshot
├── README.md
