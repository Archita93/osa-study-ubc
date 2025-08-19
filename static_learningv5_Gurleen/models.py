import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt

class EEGNetInspired(nn.Module):
    """EEGNet-inspired architecture optimized for 8-channel EEG"""
    def __init__(self, n_channels=8, n_classes=2, sequence_length=2560, 
                 dropout=0.4, kernel_length=64, F1=32, D=4, F2=64):
        super().__init__()
        
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length), 
                                     padding=(0, kernel_length//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        
        # Depthwise convolution (spatial filter)
        self.spatial_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), 
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Separable convolution
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), 
                                      padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Calculate flattened size using actual forward pass
        self.flatten_size = self._calculate_flatten_size()
        
        # Classifier
        # self.classifier = nn.Linear(self.flatten_size, n_classes)
        total_features = self.flatten_size
        self.fusion_net = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
    def _calculate_flatten_size(self):
        """Calculate the actual flattened size by doing a forward pass"""
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.zeros(1, self.n_channels, self.sequence_length)
            dummy_input = dummy_input.unsqueeze(1)  # (1, 1, channels, time)
            
            # Forward through conv layers
            x = self.temporal_conv(dummy_input)
            x = self.bn1(x)
            x = self.spatial_conv(x)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.separable_conv(x)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            
            return x.flatten(start_dim=1).shape[1]
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, channels, time)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.bn1(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Separable convolution
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Flatten and classify
        x = x.flatten(start_dim=1)
        x = self.fusion_net(x)
        
        return x


class ImprovedCNNLSTM(nn.Module):
    """Improved CNN-LSTM with better feature extraction"""
    def __init__(self, n_channels=8, n_classes=2, dropout=0.3):
        super().__init__()
        
        # Spatial-temporal feature extraction
        self.spatial_conv = nn.Conv2d(1, 8, (n_channels, 1), bias=False)
        self.temporal_conv1 = nn.Conv1d(8, 16, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.temporal_conv2 = nn.Conv1d(16, 32, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(4)
        
        self.temporal_conv3 = nn.Conv1d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(2)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(64, 32, num_layers=1, batch_first=True, 
                           bidirectional=True, dropout=0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(64, num_heads=4, 
                                             dropout=dropout, batch_first=True)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size = x.size(0)
        
        # Spatial filtering
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        x = self.spatial_conv(x)  # (batch, 8, 1, time)
        x = x.squeeze(2)  # (batch, 8, time)
        
        # Temporal convolutions
        x = F.relu(self.bn1(self.temporal_conv1(x)))
        x = F.relu(self.bn2(self.temporal_conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.temporal_conv3(x)))
        x = self.pool2(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, time, features)
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class SimpleEffectiveCNN(nn.Module):
    """Simple but effective CNN for EEG classification"""
    def __init__(self, n_channels=4, n_classes=2, dropout=0.25):
        super().__init__()
        
        # Channel-wise convolutions
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(1, 8, 25),
            self._make_conv_block(8, 16, 15),
            self._make_conv_block(16, 32, 10),
            self._make_conv_block(32, 64, 5)
        ])
        
        self.spatial_conv = nn.Conv2d(1, 4, (n_channels, 1), bias=False)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4, 32),  # 64 features * 4 spatial filters
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
        
    def _make_conv_block(self, in_ch, out_ch, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
    def forward(self, x):
        # Process each channel separately then combine
        batch_size, n_channels, seq_len = x.shape
        
        # Reshape for spatial convolution
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        spatial_features = []
        
        # Apply spatial filters
        spatial_out = self.spatial_conv(x)  # (batch, 4, 1, time)
        
        for i in range(4):  # 4 spatial filters
            channel_data = spatial_out[:, i, 0, :].unsqueeze(1)  # (batch, 1, time)
            
            # Apply temporal convolutions
            for conv_block in self.conv_blocks:
                channel_data = conv_block(channel_data)
            
            # Global pooling
            channel_feat = self.global_pool(channel_data)  # (batch, 64, 1)
            spatial_features.append(channel_feat.squeeze(-1))
        
        # Concatenate all spatial features
        x = torch.cat(spatial_features, dim=1)  # (batch, 64*4)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class SpectralFeatureExtractor:
    """Extract spectral features from EEG signals"""
    
    def __init__(self, fs=256, nperseg=256):
        self.fs = fs
        self.nperseg = nperseg
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def extract_band_powers(self, eeg_data):
        """Extract power in different frequency bands using Welch's method"""
        # eeg_data shape: (batch, channels, time)
        batch_size, n_channels, n_samples = eeg_data.shape
        
        band_powers = []
        
        # Adjust nperseg for short signals
        nperseg = min(self.nperseg, n_samples // 4) if n_samples > 8 else n_samples
        nperseg = max(nperseg, 8)  # Minimum segment size
        
        for i in range(batch_size):
            channel_powers = []
            for ch in range(n_channels):
                try:
                    signal_data = eeg_data[i, ch, :].cpu().numpy()
                    
                    # Remove DC component
                    signal_data = signal_data - np.mean(signal_data)
                    
                    # Check for constant signal
                    if np.std(signal_data) < 1e-10:
                        # Use default small powers for each band
                        ch_band_powers = [1e-6] * len(self.bands)
                    else:
                        # Compute PSD using Welch's method
                        freqs, psd = signal.welch(signal_data, 
                                                fs=self.fs, 
                                                nperseg=nperseg,
                                                noverlap=nperseg//2)
                        
                        # Extract power in each band
                        ch_band_powers = []
                        for band_name, (low, high) in self.bands.items():
                            band_mask = (freqs >= low) & (freqs <= high)
                            
                            if np.any(band_mask):
                                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                                # Ensure positive power
                                band_power = max(band_power, 1e-10)
                            else:
                                # No frequencies in this band
                                band_power = 1e-6
                            
                            ch_band_powers.append(float(band_power))
                
                except Exception as e:
                    print(f"Error computing band powers for batch {i}, channel {ch}: {e}")
                    # Use default values
                    ch_band_powers = [1e-6] * len(self.bands)
                
                channel_powers.append(ch_band_powers)
            band_powers.append(channel_powers)
        
        # Shape: (batch, channels, n_bands)
        return torch.tensor(band_powers, dtype=torch.float32)
    
    def extract_spectral_statistics(self, eeg_data):
        """Extract spectral statistics (centroid, bandwidth, etc.)"""
        batch_size, n_channels, n_samples = eeg_data.shape
        
        spectral_stats = []
        
        for i in range(batch_size):
            channel_stats = []
            for ch in range(n_channels):
                try:
                    signal_data = eeg_data[i, ch, :].cpu().numpy()
                    
                    # Remove DC component and apply window
                    signal_data = signal_data - np.mean(signal_data)
                    window = np.hanning(len(signal_data))
                    signal_data = signal_data * window
                    
                    # Compute FFT
                    fft_vals = np.abs(fft(signal_data))
                    freqs = fftfreq(len(signal_data), 1/self.fs)
                    
                    # Only use positive frequencies up to Nyquist
                    pos_mask = (freqs > 0) & (freqs <= self.fs/2)
                    fft_vals = fft_vals[pos_mask]
                    freqs = freqs[pos_mask]
                    
                    # Check if we have valid FFT values
                    if len(fft_vals) == 0 or np.sum(fft_vals) == 0:
                        # Return default values
                        centroid = 10.0  # Default to 10 Hz
                        bandwidth = 5.0
                        rolloff = 20.0
                    else:
                        # Add small epsilon to avoid division by zero
                        epsilon = 1e-10
                        fft_sum = np.sum(fft_vals) + epsilon
                        fft_vals_norm = fft_vals / fft_sum
                        
                        # Spectral centroid
                        centroid = np.sum(freqs * fft_vals_norm)
                        centroid = np.clip(centroid, 0.1, self.fs/2)  # Clip to valid range
                        
                        # Spectral bandwidth
                        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_vals_norm))
                        bandwidth = np.clip(bandwidth, 0.1, self.fs/4)
                        
                        # Spectral rolloff (95% of energy)
                        cumsum = np.cumsum(fft_vals_norm)
                        rolloff_indices = np.where(cumsum >= 0.95 * cumsum[-1])[0]
                        if len(rolloff_indices) > 0:
                            rolloff_idx = rolloff_indices[0]
                            rolloff = freqs[rolloff_idx]
                        else:
                            rolloff = freqs[-1] if len(freqs) > 0 else self.fs/2
                        
                        rolloff = np.clip(rolloff, centroid, self.fs/2)
                    
                    # Ensure no NaN or inf values
                    centroid = float(centroid) if np.isfinite(centroid) else 10.0
                    bandwidth = float(bandwidth) if np.isfinite(bandwidth) else 5.0
                    rolloff = float(rolloff) if np.isfinite(rolloff) else 20.0
                    
                    channel_stats.append([centroid, bandwidth, rolloff])
                    
                except Exception as e:
                    print(f"Error processing channel {ch} in batch {i}: {e}")
                    # Use default values on error
                    channel_stats.append([10.0, 5.0, 20.0])
                    
            spectral_stats.append(channel_stats)
        
        return torch.tensor(spectral_stats, dtype=torch.float32)
    
    def extract_wavelet_features(self, eeg_data, wavelet='db4', levels=5):
        """Extract wavelet decomposition features"""
        batch_size, n_channels, n_samples = eeg_data.shape
        
        # Adjust levels based on signal length
        max_levels = pywt.dwt_max_level(n_samples, wavelet)
        levels = min(levels, max_levels)
        levels = max(levels, 1)  # At least 1 level
        
        wavelet_features = []
        
        for i in range(batch_size):
            channel_features = []
            for ch in range(n_channels):
                try:
                    signal_data = eeg_data[i, ch, :].cpu().numpy()
                    
                    # Remove DC component
                    signal_data = signal_data - np.mean(signal_data)
                    
                    # Check for constant signal
                    if np.std(signal_data) < 1e-10:
                        # Use default features for constant signal
                        level_features = [0.0] * (4 * (levels + 1))
                    else:
                        # Wavelet decomposition
                        coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
                        
                        # Extract statistics from each level
                        level_features = []
                        for coeff in coeffs:
                            if len(coeff) > 0:
                                level_features.extend([
                                    float(np.mean(coeff)),
                                    float(np.std(coeff)),
                                    float(np.var(coeff)),
                                    float(np.max(np.abs(coeff)))
                                ])
                            else:
                                level_features.extend([0.0, 0.0, 0.0, 0.0])
                        
                        # Ensure all features are finite
                        level_features = [f if np.isfinite(f) else 0.0 for f in level_features]
                
                except Exception as e:
                    print(f"Error computing wavelet features for batch {i}, channel {ch}: {e}")
                    # Use default features
                    level_features = [0.0] * (4 * (levels + 1))
                
                channel_features.append(level_features)
            wavelet_features.append(channel_features)
        
        return torch.tensor(wavelet_features, dtype=torch.float32)

class MultiModalEEGNet(nn.Module):
    """EEGNet with integrated spectral features"""
    
    def __init__(self, n_channels=8, n_classes=2, sequence_length=2560,
                 dropout=0.4, kernel_length=64, F1=64, D=4, F2=128, fs=256):
        super().__init__()
        
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.fs = fs
        
        # Spectral feature extractor
        self.spectral_extractor = SpectralFeatureExtractor(fs=fs)
        
        # Original EEGNet layers for raw signal
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length),
                                     padding=(0, kernel_length//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        
        self.spatial_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16),
                                      padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Calculate raw signal feature size
        self.raw_feature_size = self._calculate_raw_feature_size()
        
        # Spectral feature processing
        # Band powers: 5 bands * n_channels
        self.band_power_size = 5 * n_channels
        self.band_power_net = nn.Sequential(
            nn.Linear(self.band_power_size, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Spectral statistics: 3 stats * n_channels  
        self.spectral_stats_size = 3 * n_channels
        self.spectral_stats_net = nn.Sequential(
            nn.Linear(self.spectral_stats_size, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Wavelet features: variable size based on levels
        self.wavelet_size = 4 * 6 * n_channels  # 4 stats * 6 levels * channels
        self.wavelet_net = nn.Sequential(
            nn.Linear(self.wavelet_size, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        total_features = self.raw_feature_size + 64 + 32 + 128
        self.fusion_net = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def _calculate_raw_feature_size(self):
        """Calculate the flattened size of raw EEG features"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.n_channels, self.sequence_length)
            x = dummy_input.unsqueeze(1)
            
            x = self.temporal_conv(x)
            x = self.bn1(x)
            x = self.spatial_conv(x)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.separable_conv(x)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            
            return x.flatten(start_dim=1).shape[1]
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size = x.shape[0]
        
        # Extract spectral features
        with torch.no_grad():
            band_powers = self.spectral_extractor.extract_band_powers(x)
            spectral_stats = self.spectral_extractor.extract_spectral_statistics(x)
            wavelet_features = self.spectral_extractor.extract_wavelet_features(x)
        
        # Move to same device as model
        device = next(self.parameters()).device
        band_powers = band_powers.to(device)
        spectral_stats = spectral_stats.to(device)
        wavelet_features = wavelet_features.to(device)
        
        # Process raw EEG signal through original EEGNet
        eeg_features = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        eeg_features = self.temporal_conv(eeg_features)
        eeg_features = self.bn1(eeg_features)
        eeg_features = self.spatial_conv(eeg_features)
        eeg_features = self.bn2(eeg_features)
        eeg_features = self.elu1(eeg_features)
        eeg_features = self.pool1(eeg_features)
        eeg_features = self.drop1(eeg_features)
        
        eeg_features = self.separable_conv(eeg_features)
        eeg_features = self.bn3(eeg_features)
        eeg_features = self.elu2(eeg_features)
        eeg_features = self.pool2(eeg_features)
        eeg_features = self.drop2(eeg_features)
        
        eeg_features = eeg_features.flatten(start_dim=1)
        
        # Process spectral features
        band_features = self.band_power_net(band_powers.flatten(start_dim=1))
        stats_features = self.spectral_stats_net(spectral_stats.flatten(start_dim=1))
        wavelet_feats = self.wavelet_net(wavelet_features.flatten(start_dim=1))
        
        # Concatenate all features
        combined_features = torch.cat([
            eeg_features, 
            band_features, 
            stats_features, 
            wavelet_feats
        ], dim=1)
        
        # Final classification
        output = self.fusion_net(combined_features)
        
        return output


# Model selection function
def get_model(model_name='eegnet', n_channels=4, n_classes=2, sequence_length=2560, **kwargs):
    """Factory function to get different models"""
    if model_name == 'eegnet':
        return EEGNetInspired(n_channels=n_channels, n_classes=n_classes, 
                             sequence_length=sequence_length, **kwargs)
    elif model_name == 'improved_cnn_lstm':
        return ImprovedCNNLSTM(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif model_name == 'simple_cnn':
        return SimpleEffectiveCNN(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif model_name == 'multi_modal_eegnet':
        return MultiModalEEGNet(n_channels=n_channels, n_classes=n_classes, 
                               sequence_length=sequence_length, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not found. Available: ['eegnet', 'improved_cnn_lstm', 'simple_cnn', 'multi_modal_eegnet']")
