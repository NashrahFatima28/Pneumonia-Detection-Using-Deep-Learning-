# Pneumonia-Detection-Using-Deep-Learning-
This project implements a Convolutional Neural Network (CNN) for detecting pneumonia from chest X-ray images using deep learning techniques.

📌 Project Overview
Pneumonia is a lung infection that can be life-threatening if not diagnosed early. This project uses CNN-based models to automatically detect pneumonia from chest X-rays, potentially assisting radiologists in faster and more accurate diagnosis.

🔍 Objective
Detect pneumonia from chest X-ray images using CNNs.
Achieve high accuracy and low false-negative rates.
Compare performance with different architectures (e.g., custom CNN, VGG16, ResNet50).
🧠 Technologies Used
Python
TensorFlow / Keras or PyTorch
OpenCV / PIL
NumPy / Pandas
Matplotlib / Seaborn
Scikit-learn
📂 Dataset
Dataset: Chest X-Ray Images (Pneumonia) on Kaggle

Training images: chest_xray/train/
Validation images: chest_xray/val/
Testing images: chest_xray/test/
Classes:

NORMAL
PNEUMONIA
🧱 Project Structure
requirements.txt
procfile.txt
pneumonia_model.h5.ipynb
model.h5
app.py
templates
static
🚀 How to Run
1. Install Dependencies

pip install -r requirements.txt

2. Download and Prepare Dataset

Download the dataset from Kaggle
Extract into the data/ directory
3. Train the Model

4. Evaluate the Model

📌 Result
Metric	Value
Accuracy	94.5%
Precision	93.8%
Recall	95.2%
F1 Score	94.5%
