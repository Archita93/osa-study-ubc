import numpy as np
from scipy.signal import welch, iirnotch, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dataset import PSGWindowDataset
from utils import (
    get_device, print_dataset_info, 
    print_sample_info
)

from config import DATA_CONFIG, TRAINING_CONFIG


# Notch filter at 60 Hz
def notch_filter(signal, fs=256, freq=60, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

# Band definitions
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50)
}

# Compute band power features for a single EEG sample (shape: [2, timepoints])
def compute_band_power_features(eeg_sample, fs=256):
    features = []
    for ch in eeg_sample:
        ch_filtered = notch_filter(ch, fs, freq=60)
        ch_filtered = notch_filter(ch_filtered, fs, freq=120)
        f, psd = welch(ch_filtered, fs=fs, nperseg=fs)
        for band_range in bands.values():
            idx = np.logical_and(f >= band_range[0], f <= band_range[1])
            power = np.trapz(psd[idx], f[idx])
            features.append(power)
        band_range = bands['alpha']
        idx = np.logical_and(f >= band_range[0], f <= band_range[1])
        power_alpha =  np.trapz(psd[idx], f[idx])

        band_range = bands['theta']
        idx = np.logical_and(f >= band_range[0], f <= band_range[1])
        power_theta =  np.trapz(psd[idx], f[idx])
        features.append(power_alpha / (power_theta + 1e-6))
    return features

# Build dataset of features
def extract_features_from_dataset(dataset, fs=256):
    X = []
    y = []
    for signal, label in dataset:
        features = compute_band_power_features(signal, fs=fs)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

batch_size = DATA_CONFIG['batch_size']
lr = TRAINING_CONFIG['lr']
epochs = TRAINING_CONFIG['epochs']
augment = DATA_CONFIG['use_augmentation']
augment_factor = DATA_CONFIG['augment_factor']
data_dir = DATA_CONFIG['data_dir']

# Get device
device = get_device()

print(f"=== PSG Classification Training ===")
print(f"Data directory: {data_dir}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Epochs: {epochs}")
print(f"Augmentation: {augment}")
if augment:
    print(f"Augmentation factor: {augment_factor}")
print()

# Create datasets with CORRECT normalization handling
print("Loading datasets...")

# 1. Create training dataset with normalization
train_dataset = PSGWindowDataset(
    data_dir, 
    normalize=DATA_CONFIG['normalize'],
    augment=augment,
    augment_factor=augment_factor,
    mode='train'
)

# 2. Get normalization statistics from training set
train_stats = train_dataset.get_normalization_stats()
print(f"Training normalization stats calculated for method: {train_stats['normalize']}")

# 3. Create validation dataset WITHOUT its own normalization calculation
val_dataset = PSGWindowDataset(
    data_dir,
    normalize=DATA_CONFIG['normalize'],  # Don't calculate normalization stats
    augment=False,   # Never augment validation data
    augment_factor=1,
    mode='val'
)

# 4. Apply training normalization stats to validation set
val_dataset.set_normalization_stats(train_stats)
print("Applied training normalization stats to validation set")

# Print dataset information
print_dataset_info(train_dataset, val_dataset)
print_sample_info(train_dataset)

X_train, y_train = extract_features_from_dataset(train_dataset, fs=256)
X_test, y_test = extract_features_from_dataset(val_dataset, fs=256)

# Method 1: Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
y_pred_log_train = logreg.predict(X_train)

# Method 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_train = rf.predict(X_train)

# # Method 3: MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
y_pred_mlp_train = mlp.predict(X_train)

# Print results
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_log))
print(classification_report(y_train, y_pred_log_train))
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print(classification_report(y_train, y_pred_rf_train))
print("\n=== MLP Classifier ===")
print(classification_report(y_test, y_pred_mlp))
print(classification_report(y_train, y_pred_mlp_train))