# 🙋‍♀️ About the Author
This project was developed as part of my portfolio to demonstrate proficiency in machine learning.

# 📜 License
This project is for educational purposes. Data sources remain the property of their respective owners.

# 🧠 Brain Tumor Detection from MRI Images using Convolutional Neural Network (CNN)
This project demonstrates a custom built CNN that classifies brain MRI scans as tumorous or non-tumorous. It was developed using a small dataset of 253 images, built in Google Colab, entirely using Python with TensorFlow, NumPy, OpenCV, and deployed via Gradio for interactive user testing.
Google colab link: https://drive.google.com/file/d/1MJdLVn2ar-2OGSmfFQ1jcGBtptmG7z_-/view?usp=sharing

## 📌 Project Highlights
- Built a custom CNN model from scratch using **TensorFlow/Keras**
- Preprocessed image data using **OpenCV** and **NumPy**
- Achieved **100% test accuracy** (small dataset caveat)
- Deployed a real-time prediction interface using **Gradio**
- Demonstrates tradeoff analysis between model complexity and data scale

## 🎯 Product Thinking & Tradeoffs
- Model simplicity: used shallow CNN to ensure speed and explainability
- Business value: demonstrates feasibility of using off-the-shelf tools (Gradio, TensorFlow) to rapidly prototype health tech applications.
- Scalability: easily extendable to more data or a transfer learning backbone like MobileNetV2 for higher accuracy and generalizability.

## 🗂️ Dataset Overview
- Total images used: **253**
  - **155** MRI images with brain tumors
  - **98** MRI images of healthy brains
- Images were labeled into two classes: `yes` (tumor), `no` (healthy)

- Source: provided by NTUC LearningHub course instructor

## 🔎 Data Preprocessing
- Images loaded using **OpenCV** and resized to **224x224* pixels for optimal pixel retention
- Normalized using NumPy to scale pixel values to '[0, 1]'
- Image resizing and normalization were critical to reduce compute cost while preserving classification accuracy
- Split data manually for demonstration: last 10 images as test set (for full project, use 'train_test_split')

> ⚖️ **No class balancing was applied.**
Although the dataset had more tumor images (155 vs 98), the class imbalance was not severe enough to cause biased learning. Evaluation metrics and model performance showed the classifier handled both classes reliably without oversampling or undersampling.

## 🧠 Model Architecture
Sequential CNN (Keras):
- Conv2D (32) → MaxPooling
- Conv2D (64) → MaxPooling
- Conv2D (128) → MaxPooling
- Flatten → Dense (128) + ReLU
- Dense (1) + Sigmoid

## 🧠 Compile Model
- Loss function: binary_crossentropy
- Optimizer: Adam
- Epochs trained: 10

## ✅ Model Performance
- Test accuracy: 1.0000
- Test loss: 0.0708
- Note: accuracy based on small test sample and should be interpreted as proof-of-concept, not clinical reliability.

## 📌 Limitations
- Small dataset (253 images) limits generalizability
- Grayscale images only, no contrast enhancements or 3D imaging
- Achieved high test accuracy on a small test set; future work could involve cross-validation, regularization tuning, or external validation datasets
