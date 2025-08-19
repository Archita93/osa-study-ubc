import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Replace this with your actual import
from dataset import PSGWindowDataset
from config import DATA_CONFIG, TRAINING_CONFIG

def compute_average_spectrogram(dataset, label, fs=256, nperseg=256, noverlap=64):
    """
    Compute the average spectrogram for all samples in `dataset` with the given `label`.
    Each sample must return (signal, label) where signal.shape == (n_channels, n_timepoints).
    """
    specs = []
    for signal, lbl in dataset:
        if lbl != label:
            continue
        # Compute spectrogram per channel
        ch_specs = []
        for ch in signal:
            f, t, Sxx = spectrogram(ch, fs=fs, nperseg=nperseg, noverlap=noverlap)
            # ch_specs.append(10 * np.log10(Sxx + np.finfo(float).eps))
            ch_specs.append(Sxx)
        specs.append(np.mean(ch_specs, axis=0))
    if not specs:
        raise ValueError(f"No samples found for label {label}")
    specs = np.stack(specs, axis=0)
    return f, t, specs.mean(axis=0)

def plot_average_spectrogram(f, t, spec, title, ax):
    """Plot a single average spectrogram on axes `ax`."""
    log_spec = 10 * np.log10(spec + np.finfo(float).eps)
    im = ax.pcolormesh(t, f, log_spec, shading='gouraud')
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    return im


def to_db(P):
    return 10.0 * np.log10(P + np.finfo(float).eps)

def plot_avg_and_diff_spectrograms(f0, t0, S0_lin, f1, t1, S1_lin):
    # Convert to dB for visualization
    S0_db = to_db(S0_lin)
    S1_db = to_db(S1_lin)
    diff_db = S1_db - S0_db  # proper dB difference

    # Use shared vmin/vmax for the two class plots
    vmin = min(S0_db.min(), S1_db.min())
    vmax = max(S0_db.max(), S1_db.max())

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    im0 = axes[0].pcolormesh(t0, f0, S0_db, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[0].set_title("Average Spectrogram — Class 0")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axes[0], label="Power (dB)")

    im1 = axes[1].pcolormesh(t1, f1, S1_db, shading='gouraud', vmin=vmin, vmax=vmax)
    axes[1].set_title("Average Spectrogram — Class 1")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[1], label="Power (dB)")

    # Difference: center color scale around 0
    lim = np.max(np.abs(diff_db))
    im2 = axes[2].pcolormesh(t0, f0, diff_db, shading='gouraud', vmin=-lim, vmax=+lim)
    axes[2].set_title("Difference (Class 1 − Class 0) in dB")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Frequency (Hz)")
    fig.colorbar(im2, ax=axes[2], label="Δ Power (dB)")

    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    
    fs = 256  # Sampling frequency for your EEG

    # Load your datasets
    train_dataset = PSGWindowDataset(
        DATA_CONFIG['data_dir'],
        normalize=DATA_CONFIG['normalize'],
        augment=False,
        augment_factor=1,
        mode='train'
    )

    # Compute average spectrograms for class 0 and 1
    f0, t0, spec0 = compute_average_spectrogram(train_dataset, label=0, fs=fs)
    f1, t1, spec1 = compute_average_spectrogram(train_dataset, label=1, fs=fs)

    plot_avg_and_diff_spectrograms(f0, t0, spec0, f1, t1, spec1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
