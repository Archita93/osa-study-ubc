import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


import numpy as np
import random
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split

def bandpass(sig, fs=256, low=0.5, high=45, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def multi_notch(sig, fs=256, freqs=(60, 120), Q=30):
    out = sig.copy()
    for f0 in freqs:
        b, a = iirnotch(f0, Q, fs)
        out = filtfilt(b, a, out)
    return out

# ————————————————————————————————
# 2) Sliding sub-windows
# ————————————————————————————————
def sliding_windows(sig_np, fs=256, win_s=4, step_s=2):
    npt = int(win_s * fs)
    stp = int(step_s * fs)
    T = sig_np.shape[-1]
    for start in range(0, T - npt + 1, stp):
        yield sig_np[:, start:start + npt]

# ————————————————————————————————
# 3) Augmentations (optional)
# ————————————————————————————————
def jitter_noise(win, sigma=0.01):
    return win + np.random.randn(*win.shape) * sigma

def time_shift(win, max_shift_s=0.5, fs=256):
    max_shift = int(max_shift_s * fs)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(win, shift, axis=-1)

def scale_amplitude(win, low=0.8, high=1.2):
    return win * np.random.uniform(low, high)

def augment_window(win):
    if random.random() < 0.5:
        win = jitter_noise(win, sigma=0.02)
    if random.random() < 0.5:
        win = time_shift(win, max_shift_s=0.25)
    if random.random() < 0.5:
        win = scale_amplitude(win, low=0.9, high=1.1)
    return win


class PSGWindowDataset(Dataset):
    """Dataset class for PSG window data with preprocessing and augmentation"""
    
    def __init__(self, data_dir, normalize=None, augment=False, augment_factor=2, mode='train', train_test_split_seed=42):
        """
        Args:
            data_dir: Path to single data directory containing all .npy files
            normalize: Normalization method - 'zscore', 'robust', 'minmax', or None
            augment: Whether to apply data augmentation
            augment_factor: Number of augmented samples per original sample
            mode: 'train' or 'val' (determines which split to use)
            train_test_split_seed: Random seed for reproducible train/test split
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.augment = augment
        self.augment_factor = augment_factor
        self.mode = mode
        self.train_test_split_seed = train_test_split_seed
        
        # Store file paths instead of loaded data
        self.file_paths = []
        self.labels = []
        
        # Normalization statistics (will be computed if needed)
        self.channel_means = None
        self.channel_stds = None
        self.channel_medians = None
        self.channel_mads = None
        self.channel_mins = None
        self.channel_maxs = None
        
        # Load file paths and split into train/test
        self._load_file_paths_and_split()

        self.remove_invalid_samples()
        
        # Calculate normalization statistics if needed (only for train mode)
        if self.normalize and self.mode == 'train':
            self._calculate_normalization_stats()
        
        # Generate augmented indices if requested (only for train mode)
        if augment and self.mode == 'train':
            self._generate_augmented_indices()
        
        self._print_dataset_stats()

    def notch_filter(self, signal, fs=256, freq=60, Q=30):
        b, a = iirnotch(freq, Q, fs)
        return filtfilt(b, a, signal)

    def _load_file_paths_and_split(self):
        """Load file paths and split into train/test sets"""
        all_file_paths = []
        all_labels = []
        
        # Collect all file paths and labels
        for path in glob.glob(os.path.join(self.data_dir, "*.npy")):
            if "_times" in path or "channel_names.npy" in path:
                continue
            
            label = 0 if path.endswith("_neg.npy") else 1
            
            # Quick validation check without loading full array
            try:
                arr = np.load(path)
                if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[1] < 2560:
                    continue
                all_file_paths.append(path)
                all_labels.append(label)
            except Exception as e:
                print(f"Error checking {path}: {e}")
                continue
        
        # Stratified train/test split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_file_paths, all_labels, 
            test_size=0.3, 
            random_state=self.train_test_split_seed,
            stratify=all_labels
        )
        
        # Set paths and labels based on mode
        if self.mode == 'train':
            self.file_paths = train_paths
            self.labels = train_labels
        else:  # val mode
            self.file_paths = val_paths
            self.labels = val_labels
        
        print(f"Loaded {len(self.file_paths)} file paths for {self.mode} mode")

    def remove_invalid_samples(self):
        """Remove samples with invalid shapes from the dataset"""
        valid_indices = []
        count = 0
        for idx, file_path in enumerate(self.file_paths):
            arr = np.load(file_path)
            if arr.shape[0] == 8 and arr.shape[1] == 2560:
                valid_indices.append(idx)
            else:
                count = count + 1
                # print(f"Removing invalid sample {file_path} with shape {arr.shape}")
        
        print(f"Removed {count} invalid samples from {self.mode} dataset")
        # Filter file paths and labels
        self.file_paths = [self.file_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        print(f"Remaining valid samples: {len(self.file_paths)}")

    def _load_and_process_sample(self, file_path):
        """Load and process a single sample from file path"""
        try:
            arr = np.load(file_path)
            if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[1] < 2560:
                raise ValueError(f"Invalid array shape: {arr.shape}")

            if arr.shape[0] != 8:
                return None

            m1, m2, c3, c4 = arr[0, :], arr[1, :], arr[2, :], arr[3, :]
            r = (m1 + m2)/2
            c3_ = c3 - m2
            c4_ = c4 - m1
            new_arr = np.zeros((2, arr.shape[1]))
            arr = arr[:8, :2560]
            
            return arr
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _calculate_normalization_stats(self):
        """Calculate normalization statistics based on the chosen method"""
        print(f"Calculating {self.normalize} normalization statistics...")
        
        # Collect data from a sample of files for statistics calculation
        # Use all files if dataset is small, otherwise sample
        sample_size = min(len(self.file_paths), 1000)  # Limit to 1000 samples for efficiency
        sample_indices = random.sample(range(len(self.file_paths)), sample_size)
        
        all_data = []
        for idx in tqdm(sample_indices, desc="Loading samples for normalization stats"):
            arr = self._load_and_process_sample(self.file_paths[idx])
            if arr is not None:
                if arr.shape[0] != 8:
                    continue
                all_data.append(arr)
        
        if not all_data:
            raise ValueError("No valid samples found for normalization statistics")
        
        # Stack all arrays: (n_samples, n_channels, n_timepoints)
        all_data = np.stack(all_data, axis=0)

        raw_means = np.mean(all_data, axis=(0, 2))
        raw_stds = np.std(all_data, axis=(0, 2))
        print("Raw channel means:", raw_means)
        print("Raw channel stds:", raw_stds)
        
        if self.normalize == 'zscore':
            self.channel_means = np.mean(all_data, axis=(0, 2))
            self.channel_stds = np.std(all_data, axis=(0, 2))
            self.channel_stds = np.maximum(self.channel_stds, 1e-8)
            
        elif self.normalize == 'robust':
            self.channel_medians = np.median(all_data, axis=(0, 2))
            deviations = np.abs(all_data - self.channel_medians.reshape(1, -1, 1))
            self.channel_mads = np.median(deviations, axis=(0, 2))
            self.channel_mads = np.maximum(self.channel_mads, 1e-8)
            
        elif self.normalize == 'minmax':
            self.channel_mins = np.min(all_data, axis=(0, 2))
            self.channel_maxs = np.max(all_data, axis=(0, 2))
            range_vals = self.channel_maxs - self.channel_mins
            range_vals = np.maximum(range_vals, 1e-8)
            self.channel_maxs = self.channel_mins + range_vals

    def _normalize_sample(self, arr):
        """Apply normalization to a single sample"""
        if self.normalize == 'zscore':
            return (arr - self.channel_means.reshape(-1, 1)) / self.channel_stds.reshape(-1, 1)
        
        elif self.normalize == 'robust':
            return (arr - self.channel_medians.reshape(-1, 1)) / self.channel_mads.reshape(-1, 1)
        
        elif self.normalize == 'minmax':
            return (arr - self.channel_mins.reshape(-1, 1)) / (self.channel_maxs - self.channel_mins).reshape(-1, 1)
        
        else:
            return arr

    def _generate_augmented_indices(self):
        """Generate indices for augmented samples"""
        print(f"Generating indices for {self.augment_factor} augmented versions per sample...")
        
        original_count = len(self.file_paths)
        
        # Extend file paths and labels to include augmented versions
        for i in range(original_count):
            for _ in range(self.augment_factor):
                self.file_paths.append(self.file_paths[i])  # Same file path
                self.labels.append(self.labels[i])  # Same label
        
        # Keep track of which samples are augmented
        self.is_augmented = [False] * original_count + [True] * (original_count * self.augment_factor)

    def _augment_sample(self, arr):
        """Apply multiple augmentation techniques to EEG data"""
        aug_arr = arr.copy()
        
        # 1. Gaussian noise
        if random.random() < 0.7:
            noise_std = np.std(aug_arr) * random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, aug_arr.shape)
            aug_arr = aug_arr + noise
        
        # 3. Amplitude scaling per channel
        if random.random() < 0.5:
            for ch in range(aug_arr.shape[0]):
                scale_factor = random.uniform(0.85, 1.15)
                aug_arr[ch] = aug_arr[ch] * scale_factor
        
        # 6. Frequency domain augmentation
        if random.random() < 0.3:
            for ch in range(aug_arr.shape[0]):
                fft = np.fft.fft(aug_arr[ch])
                phase_shift = random.uniform(-0.1, 0.1)
                frequencies = np.fft.fftfreq(len(fft))
                phase_shifts = np.exp(1j * 2 * np.pi * phase_shift * frequencies)
                aug_arr[ch] = np.real(np.fft.ifft(fft * phase_shifts))
        
        return aug_arr

    def _print_dataset_stats(self):
        """Print dataset statistics"""
        print(f"Total samples: {len(self.file_paths)}")
        
        class_0_count = sum(1 for l in self.labels if l == 0)
        class_1_count = sum(1 for l in self.labels if l == 1)
        print(f"Class 0: {class_0_count}, Class 1: {class_1_count}")
        
        # Print normalization info
        if self.normalize and self.mode == 'train':
            print(f"Using {self.normalize} normalization")
            if self.normalize == 'zscore' and self.channel_means is not None:
                print(f"Channel means: {self.channel_means}")
                print(f"Channel stds: {self.channel_stds}")
            elif self.normalize == 'robust' and self.channel_medians is not None:
                print(f"Channel medians: {self.channel_medians}")
                print(f"Channel MADs: {self.channel_mads}")
            elif self.normalize == 'minmax' and self.channel_mins is not None:
                print(f"Channel mins: {self.channel_mins}")
                print(f"Channel maxs: {self.channel_maxs}")

    def get_normalization_stats(self):
        """Return normalization statistics for use in validation/test sets"""
        stats = {
            'normalize': self.normalize,
            'channel_means': self.channel_means,
            'channel_stds': self.channel_stds,
            'channel_medians': self.channel_medians,
            'channel_mads': self.channel_mads,
            'channel_mins': self.channel_mins,
            'channel_maxs': self.channel_maxs
        }
        return stats
    
    def set_normalization_stats(self, stats):
        """Set normalization statistics from training set"""
        self.normalize = stats['normalize']
        self.channel_means = stats['channel_means']
        self.channel_stds = stats['channel_stds']
        self.channel_medians = stats['channel_medians']
        self.channel_mads = stats['channel_mads']
        self.channel_mins = stats['channel_mins']
        self.channel_maxs = stats['channel_maxs']

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load and return a sample at the given index"""
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and process the sample
        arr = self._load_and_process_sample(file_path)
        
        if arr is None:
            # Return a zero tensor if loading fails
            arr = np.zeros((8, 2560))
        
        # Apply augmentation if this is an augmented sample
        if hasattr(self, 'is_augmented') and self.is_augmented[idx]:
            arr = self._augment_sample(arr)
        
        # Apply normalization if enabled
        if self.normalize:
            arr = (arr - np.mean(arr, axis=1, keepdims=True)) / (np.std(arr, axis=1, keepdims=True) + 1e-6)
            # arr = self._normalize_sample(arr)
        
        x = torch.tensor(arr, dtype=torch.float32)
        return x, torch.tensor(label, dtype=torch.long)


def calculate_class_weights(dataset):
    """Calculate class weights for handling imbalanced data"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())
    
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)