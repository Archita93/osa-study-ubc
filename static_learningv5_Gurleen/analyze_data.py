import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import glob
import os

class EEGDataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.channel_names = np.load(os.path.join(data_path, "channel_names.npy"))
        self.load_all_data()
    
    def load_all_data(self):
        """Load all .npy files and organize by subject and label"""
        self.data_files = []
        self.labels = []
        self.subjects = []
        
        for npy_file in glob.glob(os.path.join(self.data_path, "*_win_*_*.npy")):
            if "times" not in npy_file:  # Skip time files
                filename = os.path.basename(npy_file)
                parts = filename.replace(".npy", "").split("_")
                subject = parts[0]
                label = parts[-1]  # pos or neg
                
                self.data_files.append(npy_file)
                self.labels.append(label)
                self.subjects.append(subject)
    
    def basic_data_stats(self):
        """Print basic statistics about the dataset"""
        labels_count = pd.Series(self.labels).value_counts()
        subjects_count = pd.Series(self.subjects).nunique()
        
        print("=== DATASET OVERVIEW ===")
        print(f"Total windows: {len(self.data_files)}")
        print(f"Unique subjects: {subjects_count}")
        print(f"Channels: {list(self.channel_names)}")
        print(f"Label distribution:")
        for label, count in labels_count.items():
            print(f"  {label}: {count} ({count/len(self.labels)*100:.1f}%)")
    
    def check_data_integrity(self, sample_size=50):
        """Check for data quality issues"""
        print("\n=== DATA INTEGRITY CHECKS ===")
        
        issues = []
        sample_files = np.random.choice(self.data_files, min(sample_size, len(self.data_files)), replace=False)
        
        shapes = []
        for file_path in sample_files:
            try:
                data = np.load(file_path)
                shapes.append(data.shape)
                
                # Check for NaN/Inf values
                if np.any(np.isnan(data)):
                    issues.append(f"NaN values in {os.path.basename(file_path)}")
                if np.any(np.isinf(data)):
                    issues.append(f"Inf values in {os.path.basename(file_path)}")
                    
                # Check for extreme values (likely artifacts)
                if np.any(np.abs(data) > 1000):  # Adjust threshold as needed
                    issues.append(f"Extreme values (>1000μV) in {os.path.basename(file_path)}")
                    
            except Exception as e:
                issues.append(f"Cannot load {os.path.basename(file_path)}: {e}")
        
        # Check shape consistency
        unique_shapes = list(set(shapes))
        print(f"Data shapes found: {unique_shapes}")
        if len(unique_shapes) > 1:
            issues.append("Inconsistent data shapes across files")
        
        if issues:
            print("ISSUES FOUND:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"{issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues)-10} more issues")
        else:
            print("No major data integrity issues found")
            
        return len(issues) == 0
    
    
    def class_balance_analysis(self, save_dir="./plots"):
        """Analyze class distribution and save 4 separate plots."""
        import os

        print("\n=== CLASS BALANCE ANALYSIS ===")

        df = pd.DataFrame({
            'subject': self.subjects,
            'label': self.labels
        })

        # Class distribution
        overall_balance = df['label'].value_counts(normalize=True)
        subject_balance = df.groupby(['subject', 'label']).size().unstack(fill_value=0)
        subject_balance['total'] = subject_balance.sum(axis=1)

        for label in ['pos', 'neg']:
            if label not in subject_balance.columns:
                subject_balance[label] = 0

        # Create output directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1. Pie chart
        plt.figure(figsize=(6, 6))
        overall_balance.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Overall Class Distribution')
        plt.ylabel('')
        plt.savefig(os.path.join(save_dir, 'class_distribution_pie.png'))
        plt.show()
        plt.close()

        # 2. Positive samples per subject
        ax = subject_balance['pos'].plot(kind='bar', color='#006400', edgecolor='black', figsize=(16, 6))
        ax.set_title('Positive Samples per Subject')
        ax.set_xlabel('Subject')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'positive_samples_per_subject.png'))
        plt.show()
        plt.close()

        # 3. Negative samples per subject
        ax = subject_balance['neg'].plot(kind='bar', color='#8B0000', edgecolor='black', figsize=(16, 6))
        ax.set_title('Negative Samples per Subject')
        ax.set_xlabel('Subject')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'negative_samples_per_subject.png'))
        plt.show()
        plt.close()

        # 4. Stacked bar per subject
        ax = subject_balance[['pos', 'neg']].plot(kind='bar', stacked=True,
                                                color=['#006400', '#8B0000'],
                                                figsize=(16, 6), edgecolor='black')
        ax.set_title('Stacked Class Distribution by Subject')
        ax.set_xlabel('Subject')
        ax.set_ylabel('Number of Samples')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        ax.legend(['Positive', 'Negative'])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'stacked_class_distribution.png'))
        plt.show()
        plt.close()

        return subject_balance


    def get_average_samples_per_subject(self):
        """Calculate average number of positive and negative samples per subject."""
        print("\n=== AVERAGE SAMPLES PER SUBJECT ===")
        
        df = pd.DataFrame({
            'subject': self.subjects,
            'label': self.labels
        })

        subject_counts = df.groupby(['subject', 'label']).size().unstack(fill_value=0)

        avg_pos = subject_counts['pos'].mean() if 'pos' in subject_counts else 0
        avg_neg = subject_counts['neg'].mean() if 'neg' in subject_counts else 0

        print(f"Average positive samples per subject: {avg_pos:.2f}")
        print(f"Average negative samples per subject: {avg_neg:.2f}")

        return avg_pos, avg_neg
    
    def sample_visualization(self, n_samples=4, save_dir="./plots"):
        """Visualize sample windows from each class"""
        print("\n=== SAMPLE VISUALIZATION ===")
        
        pos_files = [f for f, l in zip(self.data_files, self.labels) if l == 'pos']
        neg_files = [f for f, l in zip(self.data_files, self.labels) if l == 'neg']
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
         
        fig.suptitle("Sample EEG Windows: Positive vs Negative", fontsize=16)
        
        for i in range(min(n_samples, len(pos_files), len(neg_files))):
            # Positive sample
            pos_data = np.load(np.random.choice(pos_files))
            m1, m2, c3, c4 = pos_data[0, :], pos_data[1, :], pos_data[2, :], pos_data[3, :]
            r = (m1 + m2)/2
            c3_ = c3 - m2
            c4_ = c4 - m1
            new_arr = np.zeros((2, pos_data.shape[1]))
            new_arr[0, :] = c3_
            new_arr[1, :] = c4_
            pos_data = new_arr
            ax_pos = axes[i, 0] if n_samples > 1 else axes[0]
            for ch_idx, ch_name in enumerate(self.channel_names):
                if ch_idx < pos_data.shape[0]:
                    ax_pos.plot(pos_data[ch_idx, :], label=ch_name, alpha=0.8)
            ax_pos.set_title(f'Positive Sample {i+1}')
            ax_pos.set_ylabel('Amplitude (μV)')
            ax_pos.set_ylim([-6e-5, 6e-5])
            if i == 0:
                ax_pos.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Negative sample
            neg_data = np.load(np.random.choice(neg_files))
            m1, m2, c3, c4 = neg_data[0, :], neg_data[1, :], neg_data[2, :], neg_data[3, :]
            r = (m1 + m2)/2
            c3_ = c3 - m2
            c4_ = c4 - m1
            new_arr = np.zeros((2, neg_data.shape[1]))
            new_arr[0, :] = c3_
            new_arr[1, :] = c4_
            neg_data = new_arr
            ax_neg = axes[i, 1] if n_samples > 1 else axes[1]
            for ch_idx, ch_name in enumerate(self.channel_names):
                if ch_idx < neg_data.shape[0]:
                    ax_neg.plot(neg_data[ch_idx, :], label=ch_name, alpha=0.8)
            ax_neg.set_title(f'Negative Sample {i+1}')
            ax_neg.set_ylabel('Amplitude (μV)')
            ax_neg.set_ylim([-6e-5, 6e-5])
        
        plt.xlabel('Time Points')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_visuals.png'))
        plt.show()

    def find_empty_samples(self):
        """
        Scans all *_times.npy files in the given directory and prints start/end times
        for those with corresponding data arrays of shape (4, 0).
        """
        print("\n=== EMPTY SAMPLE CHECK ===")
        npy_data_dir = self.data_path
        found = False
        total_samples = 0
        for file in os.listdir(npy_data_dir):
            if file.endswith("_times.npy"):
                base_name = file.replace("_times.npy", "")
                data_file = os.path.join(npy_data_dir, f"{base_name}.npy")
                times_file = os.path.join(npy_data_dir, file)

                try:
                    data = np.load(data_file)
                    if data.shape == (4, 0):
                        times = np.load(times_file)
                        total_samples +=1 
                        if len(times) >= 2:
                            start_time, end_time = times[0], times[-1]
                        else:
                            start_time = end_time = "Unknown"
                        print(f"Empty sample: {base_name} | Start: {start_time} | End: {end_time}")
                        found = True
                except Exception as e:
                    print(f"Error loading {base_name}: {e}")
        
        print(f"Total samples are {total_samples}")
        if not found:
            print("No empty (4, 0) samples found.")

    def band_power_comparison(self, sampling_rate=256, n_samples=50):
        """Compare frequency band power between positive and negative windows"""
        print("\n=== BAND POWER COMPARISON (POS vs NEG) ===")
        
        # EEG frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        
        # Separate files by class
        pos_files = [f for f, l in zip(self.data_files, self.labels) if l == 'pos']
        neg_files = [f for f, l in zip(self.data_files, self.labels) if l == 'neg']
        
        pos_sample = np.random.choice(pos_files, min(n_samples//2, len(pos_files)), replace=False)
        neg_sample = np.random.choice(neg_files, min(n_samples//2, len(neg_files)), replace=False)
        
        # Calculate band powers
        band_powers = {
            'pos': {band: {ch: [] for ch in self.channel_names} for band in bands},
            'neg': {band: {ch: [] for ch in self.channel_names} for band in bands}
        }
        def calculate_band_power(data, fs, band_range):
                """Calculate power in a specific frequency band"""
                f, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)//4))
                
                # Find frequency indices for the band
                band_mask = (f >= band_range[0]) & (f <= band_range[1])
                band_power = np.trapz(psd[band_mask], f[band_mask])
                
                return band_power
        
        # Process positive samples
        for file_path in pos_sample:
            data = np.load(file_path)
            for ch_idx, ch_name in enumerate(self.channel_names):
                if ch_idx < data.shape[0]:
                    ch_data = data[ch_idx, :]
                    for band_name, band_range in bands.items():
                        power = calculate_band_power(ch_data, sampling_rate, band_range)
                        band_powers['pos'][band_name][ch_name].append(power)
        
        # Process negative samples
        for file_path in neg_sample:
            data = np.load(file_path)
            for ch_idx, ch_name in enumerate(self.channel_names):
                if ch_idx < data.shape[0]:
                    ch_data = data[ch_idx, :]
                    for band_name, band_range in bands.items():
                        power = calculate_band_power(ch_data, sampling_rate, band_range)
                        band_powers['neg'][band_name][ch_name].append(power)
        
        # Statistical comparison and visualization
        fig, axes = plt.subplots(len(bands), len(self.channel_names), 
                                figsize=(4*len(self.channel_names), 3*len(bands)))
        if len(self.channel_names) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle("Band Power Comparison: Positive vs Negative Windows", fontsize=16)
        
        print("Band Power Statistics (μV²/Hz):")
        
        for band_idx, (band_name, band_range) in enumerate(bands.items()):
            print(f"\n{band_name} Band ({band_range[0]}-{band_range[1]} Hz):")
            
            for ch_idx, ch_name in enumerate(self.channel_names):
                pos_powers = band_powers['pos'][band_name][ch_name]
                neg_powers = band_powers['neg'][band_name][ch_name]
                
                if pos_powers and neg_powers:
                    # Statistics
                    pos_mean = np.mean(pos_powers)
                    neg_mean = np.mean(neg_powers)
                    pos_std = np.std(pos_powers)
                    neg_std = np.std(neg_powers)
                    
                    print(f"  {ch_name}: POS={pos_mean:.2e}±{pos_std:.2e}, NEG={neg_mean:.2e}±{neg_std:.2e}")
                    
                    # Statistical test (Mann-Whitney U test for non-parametric comparison)
                    from scipy.stats import mannwhitneyu
                    try:
                        statistic, p_value = mannwhitneyu(pos_powers, neg_powers, alternative='two-sided')
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        print(f"         p-value: {p_value:.4f} {significance}")
                    except:
                        print(f"         p-value: Could not compute")
                    
                    # Plotting
                    ax = axes[band_idx, ch_idx]
                    
                    # Box plots
                    bp = ax.boxplot([pos_powers, neg_powers], 
                                labels=['Positive', 'Negative'],
                                patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightgreen')
                    bp['boxes'][1].set_facecolor('lightcoral') 
                    bp['boxes'][0].set_alpha(0.7)
                    bp['boxes'][1].set_alpha(0.7)
                    
                    ax.set_title(f'{ch_name} - {band_name}')
                    ax.set_ylabel('Power (μV²/Hz)')
                    ax.set_yscale('log')  # Log scale for better visualization
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return band_powers
    
    def amplitude_statistics_per_channel(self, n_samples=100, save_dir = "./plots"):
        """Analyze amplitude statistics for each channel across all samples"""
        print("\n=== AMPLITUDE STATISTICS PER CHANNEL ===")
        
        sample_files = np.random.choice(self.data_files, min(n_samples, len(self.data_files)), replace=False)
        
        # Collect amplitude stats per channel
        channel_stats = {ch: {'mean': [], 'median': [], 'std': [], 'min': [], 'max': [], 'ptp': []} 
                        for ch in self.channel_names}
        
        for file_path in sample_files:
            data = np.load(file_path)  # Shape: (n_channels, n_timepoints)
            
            for ch_idx, ch_name in enumerate(self.channel_names):
                if ch_idx < data.shape[0]:
                    ch_data = data[ch_idx, :]
                    channel_stats[ch_name]['mean'].append(np.mean(ch_data))
                    channel_stats[ch_name]['median'].append(np.median(ch_data))
                    channel_stats[ch_name]['std'].append(np.std(ch_data))
                    channel_stats[ch_name]['min'].append(np.min(ch_data))
                    channel_stats[ch_name]['max'].append(np.max(ch_data))
                    channel_stats[ch_name]['ptp'].append(np.ptp(ch_data))  # peak-to-peak
        
        # Create summary statistics
        summary_stats = {}
        for ch_name in self.channel_names:
            if channel_stats[ch_name]['mean']:  # Check if channel has data
                summary_stats[ch_name] = {
                    'mean_amplitude': np.mean(channel_stats[ch_name]['mean']),
                    'std_amplitude': np.mean(channel_stats[ch_name]['std']),
                    'median_amplitude': np.mean(channel_stats[ch_name]['median']),
                    'min_amplitude': np.mean(channel_stats[ch_name]['min']),
                    'max_amplitude': np.mean(channel_stats[ch_name]['max']),
                    'avg_peak_to_peak': np.mean(channel_stats[ch_name]['ptp'])
                }
        
        # Print summary
        print(f"Channel amplitude statistics (across {len(sample_files)} samples):")
        for ch_name, stats in summary_stats.items():
            print(f"\n{ch_name}:")
            print(f"  Mean amplitude: {stats['mean_amplitude']:>12.8f} μV")
            print(f"  Std amplitude:  {stats['std_amplitude']:>12.8f} μV")
            print(f"  Peak-to-peak:   {stats['avg_peak_to_peak']:>12.8f} μV")
            print(f"  Range: [{stats['min_amplitude']:>12.8f}, {stats['max_amplitude']:>12.8f}] μV")
        
        # Plot amplitude statistics
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Amplitude Statistics per Channel", fontsize=16)
        
        metrics = ['mean', 'median', 'std', 'min', 'max', 'ptp']
        metric_labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Peak-to-Peak']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i//3, i%3]
            
            # Box plot for each channel
            data_to_plot = [channel_stats[ch][metric] for ch in self.channel_names 
                           if channel_stats[ch][metric]]
            channel_labels = [ch for ch in self.channel_names if channel_stats[ch][metric]]
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=channel_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{label} Amplitude (μV)')
                ax.set_ylabel('Amplitude (μV)')
                ax.grid(True, alpha=0.3)
                
                # Flag potential issues
                if metric == 'std':
                    # Very high or very low std might indicate issues
                    for j, ch_data in enumerate(data_to_plot):
                        avg_std = np.mean(ch_data)
                        if avg_std > 100:  # High noise threshold
                            ax.text(j+1, avg_std, ' ', ha='center', va='bottom', fontsize=12)
                        elif avg_std < 1:  # Very flat signal
                            ax.text(j+1, avg_std, ' ', ha='center', va='top', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,'amplitude_stats.png'))
        
        return summary_stats

    def run_full_analysis(self):
        """Run all analyses"""
        self.basic_data_stats()
        
        if self.check_data_integrity():
            balance_df = self.class_balance_analysis()
            self.get_average_samples_per_subject()
            self.amplitude_statistics_per_channel()
            self.sample_visualization()
            
            print("\n=== RECOMMENDATIONS FOR DL TRAINING ===")
            
            # Data quality recommendations
            total_samples = len(self.data_files)
            pos_ratio = self.labels.count('pos') / total_samples
            
            if pos_ratio < 0.3 or pos_ratio > 0.7:
                print("Consider class balancing techniques (SMOTE, class weights, etc.)")
            
            if total_samples < 1000:
                print("Small dataset - consider data augmentation or transfer learning")
            
            print("Suggested preprocessing steps:")
            print("   - Standardization/normalization per channel")
            print("   - Artifact removal (if not done)")
            print("   - Consider filtering (e.g., 0.5-30 Hz bandpass)")
            print("   - Cross-validation with subject-aware splits")
            
            return balance_df
        else:
            print("Fix data integrity issues before proceeding with analysis")
            self.find_empty_samples()
            return None



# Usage example:
analyzer = EEGDataAnalyzer(r"/path/to/your/data")
balance_df = analyzer.run_full_analysis()