# 🙋‍♀️ About the Author
This project was developed as part of my portfolio to demonstrate proficiency in machine learning.

# 📜 License
This project is for educational purposes. Dataset belongs to its original creator on Kaggle.

# 🐱🐶 Cat vs Dog Image Classification using Transfer Learning
This project demonstrates a production-relevant image classifier that distinguishes between cat and dog images using **transfer learning with MobileNetV2**.
Built on Kaggle dataset of almost 25,000 images, this project explores traditional CNN models against pre-trained networks and showcases fine-tuning for performance opization. Final deployment is done via a Gradio interface for interactive, real-time classification.
Developed in Kaggle: [View Notebook](https://www.kaggle.com/code/jezzie/binary-classification-dog-cat-using-transfer-learn)

## 📌 Project Highlights
- Built and evaluated both **custom CNN** and **MobileNetV2 transfer learning** models
- Demonstrated **feature extraction vs. fine-tuning tradeoffs**
- Preprocessed data using **TensorFlow ImageDataGenerator**, avoiding memory crashes from raw OpenCV loading
- Achieved **98.48% validation accuracy** using a fine-tuned MobileNetV2 model
- Deployed via **Gradio** to enable hands-on testing by non-technical users

## 🎯 Product Thinking
- Business value: demonstrates how transfer learning can reduce training time and improve model performance in consumer-grade image classification
- Technical tradeoffs: balanced between model accuracy and training cost by comparing CNNs versus pre-trained models
- User-centric: final product includes Gradio interface for demo and testing
- Scalability: model structure and data streaming pipeline are ready for production-scale datasets

## 🗂️ Dataset Overview
- Source: [Kaggle – Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- Total images: **24,998**
  - **12,499 cat images**
  - **12,499 dog images**
- Images are RGB JPEGs with varying dimensions
- Balanced dataset removes the need for balancing

## 🔄 Data Preprocessing

### Initial Attempts:
- Attempted image loading with **OpenCV** (`cv2`), but Colab/Kaggle crashed due to memory constraints
- Switched to **Keras ImageDataGenerator**, enabling:
  - Efficient **streaming batch loading**
  - Real-time **data augmentation** (rotation, zoom, flipping)
  - **Train-test split** (80/20)

### Final Workflow:
- Resized all images to **224x224 pixels** (compatible with MobileNetV2)
- Normalized pixel values using `preprocess_input()` from **TensorFlow Keras Applications**
- Augmentation applied during training improved generalization

## 🧪 Modeling Approaches and Trade-offs
### 1. 🧱 Baseline CNNs
- CNN-3 model: 3 convolution layers → **79.77% accuracy**
- CNN-4 model: 4 convolution layers → **86.17% accuracy**
- Limitations:
  - Required manual tuning
  - Slower to converge
  - Risk of overfitting without large-scale regularization

### 2. ✅ Transfer Learning with MobileNetV2
#### Phase 1: Feature Extraction (Base model frozen)
- Loaded MobileNetV2 from TensorFlow Hub with weights frozen
- Added:
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(1, activation='sigmoid')
- Accuracy: **98.48%**
- Fast convergence with minimal tuning
- This setup already performed exceptionally well and would have been sufficient for most production needs

#### Phase 2: Fine-Tuning (Last 30 layers unfrozen)
- Unfroze **last 30 layers** of MobileNetV2
- Recompiled with **Adam optimizer (lr=1e-5)**
- Trained for a few additional epochs to allow fine-tuning of deeper features
- Accuracy: **98.54%**
- Fine-tuning was not required to reach high accuracy but demonstrated control over deep learning model behaviour and optimization

## 🧠 Final Model Architecture
Base: MobileNetV2 (partial fine-tuning)
- GlobalAveragePooling2D
- Dropout(0.3)
- Dense(1, activation="sigmoid")

## 🧠 Compile Model
- Loss: binary_crossentropy
- Optimizer: Adam
- Epochs: 10 (frozen) + 3 (fine-tuning)
- Batch size: 64

## ✅ Model Performance
- CNN 3 layers: 79.77% accuracy, simple and limited model
- CNN 4 layers: 86.17% accuracy, improved but still suboptimal
- MobileNetV2 (frozen): 98.48% accuracy, high accuracy and fast training
- MobileNetV2 (fine-tuned): 98.54%, slight improvement from MobileNetV2 (frozen)
- Transfer learning with MobileNetV2 achieved the highest accuracy of 98.48% without having to manually design CNN architecture.

