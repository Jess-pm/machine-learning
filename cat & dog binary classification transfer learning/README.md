# 🙋‍♀️ About the Author
This project was developed as part of my portfolio to demonstrate proficiency in machine learning.

# 📜 License
This project is for educational purposes. Dataset belongs to its original creator on Kaggle.

# 🐱🐶 Cat vs Dog Image Classification using Transfer Learning
This project demonstrates a **binary image classification** model that distinguishes between cat and dog images using **transfer learning with MobileNetV2**.
Built on Kaggle dataset with nearly 25,000 images, this project explores multiple modeling approaches from traditional machine learning models to deep learning, including model fine-tuning and deployed via a Gradio interface.
Developed in Kaggle: [View Notebook](https://www.kaggle.com/code/jezzie/binary-classification-dog-cat-using-transfer-learn)

## 📌 Project Summary
- Explored two modeling approaches:
  1. Convolutional Neural Network (CNN)
  2. Transfer learning using MobileNetV2 (with and without fine-tuning)
- Dataset: 24,998 color images (12,499 cats, 12,499 dogs)
- Deployed using Gradio for interactive testing

## 🗂️ Dataset Overview
- Source: [Kaggle – Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- Total images: **24,998**
  - **12,499 cat images**
  - **12,499 dog images**
- Balanced dataset removes the need for resampling techniques

## 🔄 Data Preprocessing

### For CNN:
- Resized all images to **224x224 pixels**
- Normalized pixel values to the [0, 1] range
- Used Keras' ImageDataGenerator for:
  - Directory-based loading and labeling
  - Batch-wise streaming to manage memory
  - On-the-fly augmentation (rotation, zoom, flip)
  - Attempted loading all images using cv2, but this caused session crashes due to memory overflow in both Colab and Kaggle.

### For Deep Learning:
- Resized images to **224x224 pixels**
- Preprocessed using MobileNetV2's built-in 'preprocess_input()' for scaling
- TensorFlow's ImageDataGenerator used for:
  - Real-time data augmentation
  - Efficient batch loading
  - 80/20 train-validation split
  - Attempted loading all images using cv2, but this caused session crashes due to memory overflow in both Colab and Kaggle. ImageDataGenerator solved this by streaming data in batches.

## 🧪 Modeling Approaches and Trade-offs

### 1. 📉 Logistic Regression (Baseline)
- Started with 3 convolutional layers, achieved an accuracy of **79.77%**
- Increased to 4 convolutional layers, achieved an accuracy of **86.17%**

### 2. ✅ Transfer Learning with MobileNetV2

#### Phase 1: Feature Extraction (Base model frozen)
- Used MobileNetV2 with all layers frozen
- Added:
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(1, activation='sigmoid')
- Accuracy: **98.48%**
- This setup already performed exceptionally well and would have been sufficient for most production needs

#### Phase 2: Fine-Tuning (Last 30 layers unfrozen)
- Unfroze the **last 30 layers** of MobileNetV2
- Recompiled the model with a **lower learning rate (1e-5)** to avoid large weight updates
- Trained for a few additional epochs to allow fine-tuning of deeper features
> 🎯 **Fine-tuning impact**:  
- Slight improvement in validation accuracy to **98.54%**
- Included to demonstrate the technical process of fine-tuning, not because it was required to improve performance
- Validated knowledge of layer freezing, learning rate management, and post-tuning evaluation

## 🧠 Final Model Architecture
Base: MobileNetV2 (partial fine-tuning)
Top Layers:
- GlobalAveragePooling2D
- Dropout(0.3)
- Dense(1, activation="sigmoid")

## 🧠 Compile Model
- Loss: binary_crossentropy
- Optimizer: Adam
- Epochs: 10 (frozen) + 3 (fine-tuning)
- Batch size: 64

## ✅ Model Performance
Transfer learning with MobileNetV2 achieved the highest accuracy of 98.48% without having to manually design CNN architecture.
