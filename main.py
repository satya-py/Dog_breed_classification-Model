

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import islice

# --- New Imports for Feature Engineering and ML ---
from skimage import feature  # For LBP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---

TRAIN_DIR = "train" 
LABELS_FILE = "labels.csv" 

# 1. Feature Extraction Functions ---

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """Extracts a 3D HSV color histogram from an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the 3D color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])

    # Normalize the histogram and flatten it
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp(image_path, num_points=24, radius=8):
    """Extracts a Local Binary Pattern (LBP) texture histogram from an image."""
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    # Calculate the LBP representation
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")

    # Create a histogram of the LBP pattern
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7) 

    return hist

def extract_combined_features(image_path):
    """Combines Color Histogram and LBP features."""
    color_feat = extract_color_histogram(image_path)
    lbp_feat = extract_lbp(image_path)

    if color_feat is not None and lbp_feat is not None:
        # Concatenate the features to create a single feature vector
        return np.concatenate([color_feat, lbp_feat])
    return None

# --- 2. Main Data Processing and Training Pipeline ---

def run_classification_pipeline():
    """Loads data, extracts features, trains the model, and evaluates accuracy."""
    print("--- Starting Dog Breed Classification Pipeline ---")

    # Load labels data
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"Total number of samples found: {len(labels_df)}")

    features = []
    targets = []

    # Process all data
    print("Extracting combined LBP and Color features (This may take a few minutes)...")
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        img_id = row['id']
        breed = row['breed']
        img_path = os.path.join(TRAIN_DIR, img_id + ".jpg")

        feat = extract_combined_features(img_path)
        
        if feat is not None:
            features.append(feat)
            targets.append(breed)

    # Convert to numpy arrays
    features = np.array(features)
    targets = np.array(targets)
    print(f"Feature matrix shape (Samples x Features): {features.shape}")

    # --- Preprocessing ---
    
    # 1. Scaling the Features
    scaler = StandardScaler()
    # Fit and transform the features
    features_scaled = scaler.fit_transform(features) 
    print("Features scaled successfully using StandardScaler.")

    # 2. Splitting the Scaled Data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled,  # Use the SCALED features
        targets, 
        test_size=0.2, 
        random_state=42,
        stratify=targets  # Important: Ensures equal breed proportions in train/test sets
    )
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # --- Model Training ---

 
    # n_estimators=500 will give better results than 100, but take longer to train.
    model = RandomForestClassifier(
        n_estimators=500, 
        random_state=42, 
        n_jobs=-1,  
        max_depth=20 
    )

    print("Training the RandomForestClassifier (n_estimators=500)...")
    model.fit(X_train, y_train)

    # --- Evaluation ---

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the final accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Final Results ---")
    print(f"Model: RandomForestClassifier")
    print(f"Feature Set: LBP (Texture) + Color Histogram")
    print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_classification_pipeline()