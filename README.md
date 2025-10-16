# Dog Breed Classification Project (Infosys Internship)

## 🎯 Project Goal
The primary objective of this project is to build and evaluate a robust model for classifying 120 different dog breeds, adhering strictly to traditional Machine Learning (ML) techniques (excluding CNNs/Deep Learning).

## 🚀 Milestones Achieved

| Milestone | Status | Description |
| :--- | :--- | :--- |
| **1. Data Preparation & Clean** | ✅ **Completed** | Loaded and validated the `labels.csv` file. Established file paths to the `train/` directory. Verified and handled missing or corrupted image files. |
| **2. Simple ML Model Training** | ✅ **Completed** | Implemented feature extraction using **Color Histograms** and trained a baseline **Random Forest Classifier** to establish initial accuracy. |
| **3. Advanced Feature Engineering & Tuning** | 🚧 *In Progress* | Integrate HOG and LBP features, apply aggressive hyperparameter tuning, and implement class balancing to maximize model accuracy within traditional ML constraints. |
| **4. Final Evaluation & Documentation** | ⬜ *Pending* | Finalize model training, generate detailed metrics (Confusion Matrix, Classification Report), and complete project documentation. |

---

## 🛠️ Milestone 2 Implementation Details

### Model: Simple Random Forest Classifier

* **Objective:** To establish a baseline accuracy against which subsequent feature engineering improvements can be measured.
* **Feature Used:** Only **Color Histogram (HSV)** was extracted.
* **Model Parameters:** `RandomForestClassifier(n_estimators=100)`.
* **Baseline Accuracy:** [Insert the low baseline accuracy here, e.g., ~5.04%]. This low figure is expected due to the simplicity of the features for a complex multi-class problem (120 classes).