#!/usr/bin/env python3
"""
Main training script for PSG classification
"""

import argparse
import sys
from pathlib import Path
import config

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
from dataset import PSGWindowDataset, calculate_class_weights
from train import train_model, evaluate_model, save_training_history
from utils import (
    get_device, create_data_loaders, print_dataset_info, 
    print_sample_info, plot_training_curves
)

from sklearn.metrics import classification_report, accuracy_score

from config import DATA_CONFIG, TRAINING_CONFIG

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from models import get_model

def average_psd_across_class(samples, labels, class_id, fs=256):
    psds = []
    for arr, label in zip(samples, labels):
        if label != class_id:
            continue
        f, psd_c1 = welch(arr[0], fs=fs, nperseg=fs)
        f, psd_c2 = welch(arr[1], fs=fs, nperseg=fs)
        # avg_psd = psd_c1
        avg_psd = (psd_c1 + psd_c2) / 2
        psds.append(avg_psd)
    
    mean_psd = np.mean(psds, axis=0)
    return f, mean_psd


def plot_psd_two_classes_grid(
    class0, class1, fs=256, channel_names=None, channels=None, nperseg=None, freq_xlim=None
):
    """
    Plot average Welch PSDs for two classes in a 2x7 grid.

    Parameters
    ----------
    class0 : np.ndarray, shape (N0, C, T)
        Signals for class 0 (N0 samples, C channels, T timepoints).
    class1 : np.ndarray, shape (N1, C, T)
        Signals for class 1.
    fs : float
        Sampling frequency (Hz).
    channel_names : list[str] or None
        Names for channels; length must be >= C if provided.
    channels : list[int] or None
        Exactly 7 channel indices to plot (e.g., [0,1,2,3,4,5,6]).
        If None, the first 7 channels are used (or all if C<7).
    nperseg : int or None
        Segment length for Welch. Defaults to fs (≈1 s windows).
    freq_xlim : tuple(float,float) or None
        If set, x-axis limits in Hz, e.g., (0, 50).
    """
    assert class0.ndim == 3 and class1.ndim == 3, "Inputs must be (N, C, T)"
    _, C, _ = class0.shape
    assert class1.shape[1] == C, "Both classes must have same channel count"

    if channels is None:
        channels = list(range(min(8, C)))
    else:
        assert len(channels) == 8, "Provide exactly 7 channel indices"

    if nperseg is None:
        nperseg = int(fs)

    def avg_welch_over_samples(Xch):
        # Xch: array of shape (N, T) for a single channel
        f_ref = None
        psds = []
        for x in Xch:
            f_i, pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant', scaling='density')
            if f_ref is None:
                f_ref = f_i
            elif not np.array_equal(f_i, f_ref):
                # Defensive: interpolate if grids differ
                pxx = np.interp(f_ref, f_i, pxx)
            psds.append(pxx)
        return f_ref, np.mean(psds, axis=0) if psds else (f_ref, None)

    fig, axes = plt.subplots(1, 8, figsize=(22, 6), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for col, ch in enumerate(channels):
        # Class 0
        f0, psd0 = avg_welch_over_samples(class0[:, ch, :])
        axes[0, col].semilogy(f0, psd0)
        ch_name = (channel_names[ch] if channel_names is not None else f"Ch {ch}")
        axes[0, col].set_title(f"{ch_name} — Class 0")
        axes[0, col].grid(True)

        # Class 1
        f1, psd1 = avg_welch_over_samples(class1[:, ch, :])
        axes[0, col].semilogy(f1, psd1)
        axes[0, col].set_title(f"{ch_name} — Class 1")
        axes[0, col].grid(True)

        if freq_xlim is not None:
            axes[0, col].set_xlim(*freq_xlim)
            axes[1, col].set_xlim(*freq_xlim)

    # Labels on the outer edges only
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD (µV²/Hz)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")

    fig.suptitle("Average PSDs by Class and Channel", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_psd(eeg_signal, fs=256, channel_names=None, label=None):
    """
    Plots the Power Spectral Density (PSD) of a multichannel EEG signal using Welch's method.

    Parameters:
    - eeg_signal: np.array of shape (n_channels, n_timepoints)
    - fs: Sampling frequency (Hz)
    - channel_names: Optional list of channel names for labeling
    - label: Optional class label (e.g., 0 or 1) for title
    """
    n_channels = eeg_signal.shape[0]
    plt.figure(figsize=(10, 4 * n_channels))

    for ch in range(n_channels):
        f, psd = welch(eeg_signal[ch], fs=fs, nperseg=fs)

        plt.subplot(n_channels, 1, ch + 1)
        plt.semilogy(f, psd)
        ch_name = channel_names[ch] if channel_names else f'Channel {ch}'
        plt.title(f'PSD - {ch_name}' + (f' (Label: {label})' if label is not None else ''))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (uV^2/Hz)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_sklearn_model(model, train_dataset, val_dataset):
    # Convert datasets to numpy arrays
    X_train = [sample[0].numpy().flatten() for sample in train_dataset]
    y_train = [sample[1] for sample in train_dataset]
    X_val = [sample[0].numpy().flatten() for sample in val_dataset]
    y_val = [sample[1] for sample in val_dataset]

    # Train the sklearn model
    print(f"\nTraining sklearn model: {model}")
    model.fit(X_train, y_train)

    # Evaluate the model
    print("\nEvaluating sklearn model...")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")


def main():

    model = get_model('simple_cnn', n_channels=8, n_classes=2, dropout=0.25)
    
    # Get parameters
    batch_size = DATA_CONFIG['batch_size']
    lr = TRAINING_CONFIG['lr']
    epochs = TRAINING_CONFIG['epochs']
    augment = DATA_CONFIG['use_augmentation']
    augment_factor = DATA_CONFIG['augment_factor']
    data_dir = DATA_CONFIG['data_dir']
    
    # Get device
    device = get_device()
    model.to(device)
    
    print(f"=== PSG Classification Training ===")
    print(f"Model: {model}")
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
    val_stats = val_dataset.get_normalization_stats()
    val_dataset.set_normalization_stats(val_stats)
    print("Applied training normalization stats to validation set")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: No valid samples found in datasets!")
        return
    
    # Print dataset information
    print_dataset_info(train_dataset, val_dataset)
    print_sample_info(train_dataset)
    
    # Verify normalization is working correctly
    print("\nVerifying normalization consistency:")
    train_sample, _ = train_dataset[0]
    val_sample, _ = val_dataset[0]
    print(f"Train sample - Mean: {train_sample.mean():.3f}, Std: {train_sample.std():.3f}")
    print(f"Val sample - Mean: {val_sample.mean():.3f}, Std: {val_sample.std():.3f}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")


    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, 
        batch_size=batch_size,
        num_workers=DATA_CONFIG['num_workers']
    )
    
    # Prepare training configuration
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        'epochs': epochs,
        'lr': lr,
        'class_weights': class_weights
    })
    

    # # Train model
    print(f"\nStarting training...")
    history = train_model(model, train_loader, val_loader, training_config, device)
    
    # # Final evaluation
    print(f"\nFinal evaluation on validation set:")
    # load best model if available
    if Path(TRAINING_CONFIG['model_save_path']).exists():
        model.load_state_dict(torch.load(TRAINING_CONFIG['model_save_path'], map_location=device))
    evaluate_model(model, val_loader, device)
    
    # Plot training curves
    print(f"\nPlotting training curves...")
    plot_training_curves(history)
    save_training_history(history)



if __name__ == "__main__":
    main()