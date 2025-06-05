# 🙋‍♀️ About the Author
This project was developed as part of my portfolio to demonstrate proficiency in machine learning.

# 📜 License
This project is for educational purposes. Data sources remain the property of their respective owners.

# 🧠 Brain Tumor Detection from MRI Images using Convolutional Neural Network (CNN)
This project demonstrates the use of a CNN to classify brain MRI scans as either **has tumour** or **no tumour**. It was developed as a personal machine learning portfolio project using a dataset of **253 images** (155 with brain tumors, 98 without), trained in Google Colab, and deployed via a Gradio interface for interactive testing.
Google colab link: https://drive.google.com/file/d/1MJdLVn2ar-2OGSmfFQ1jcGBtptmG7z_-/view?usp=sharing

## 📌 Project Highlights
- Trained a binary image classifier using CNN on grayscale brain MRI scans
- Achieved 100% test accuracy on the evaluation set (see limitations below)
- Deployed with a user-friendly **Gradio** interface for live image classification
- Built and tested entirely in **Google Colab**

## 🗂️ Dataset Overview
- Total images used: **253**
  - **155** MRI images with brain tumors
  - **98** MRI images of healthy brains
- Images were labeled into two classes: `yes` (tumor), `no` (healthy)
- Source: provided by NTUC LearningHub course instructor

## 🔎 Data Preprocessing
- All images were:
  - Resized to **224x224 pixels**
  - Converted to **grayscale**
  - Normalized to scale pixel values between 0 and 1

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
