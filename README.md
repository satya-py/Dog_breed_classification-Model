# Dog Breed Classification using Traditional Machine Learning

## 📝 Project Overview

This repository contains a solution for the Dog Breed Identification challenge, which requires classifying images into **120 different dog breeds**.

**Crucially, this solution is constrained to use traditional Machine Learning (ML) techniques and feature engineering, explicitly avoiding Deep Learning methods like Convolutional Neural Networks (CNNs) and pre-trained models.**

Given the complexity of the task (120 classes) and the limitations of traditional ML on image data, the approach focuses on maximizing feature quality and tuning a robust classifier.

## 🛠️ Methodology & Features

The model uses a combination of classic computer vision techniques to extract a powerful feature vector from each image:

1.  **HOG (Histogram of Oriented Gradients):** Captures **local shape and contour** information, which is excellent for defining the structure and outlines of the dog.
2.  **LBP (Local Binary Patterns):** Captures **texture information** (e.g., fur coarseness, coat patterns).
3.  **Color Histogram (HSV):** Captures the distribution of **color** across the image.

These three feature vectors are concatenated and scaled using `StandardScaler` to create the final input for the classification model.

### Classification Model

The classification is performed using a **Random Forest Classifier**, which is highly effective on high-dimensional feature vectors. The model was specifically tuned to handle the multi-class and potentially imbalanced nature of the dataset:

* **`n_estimators=1000`**: Increased number of trees for better robustness and generalization.
* **`class_weight='balanced'`**: Ensures that the model fairly weights the loss contributions from all 120 classes, preventing it from favoring the more frequent breeds.

## 🚀 Setup and Installation

### Prerequisites

* Python 3.x
* The necessary dataset (images in `train/` folder and `labels.csv`).

### Installation

Clone the repository and install the required Python libraries:

```bash
git clone [https://github.com/satya-py/Dog_breed_classification-Model.git](https://github.com/satya-py/Dog_breed_classification-Model.git)
cd Dog_breed_classification-Model
pip install pandas numpy opencv-python scikit-learn scikit-image tqdm