# PSG EEG Hypopnea Forecasting

This repository contains a small toolkit to **prepare PSG/EEG windows**, **analyze the generated dataset**, and **train/evaluate models** (both deep learning and classical ML). It also includes utilities to **visualize spectrograms** and **inspect PSDs**.

> **Files covered**: `config.py`, `main.py`, `analyze_data.py` (aka `data_analyzer.py`), `labelling_samples_v2.py` (aka `data_generation.py`), `train_sklearn.py`, and `spectrogram_viz.py`.

---

## 1) Quick Start

```bash
# 1) Create a virtual environment (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt  # create this from your imports if not present

# 3) Configure paths & params
#    Edit DATA_CONFIG and TRAINING_CONFIG in config.py

# 4) Generate (or point to) windowed .npy data
#    See §4 Data generation / labeling pipeline

# 5) Explore & sanity‑check your data
python analyze_data.py --data-dir <PATH_TO_NPY_DIR>

# 6) Train a PyTorch model (main deep-learning entrypoint)
python main.py

# 7) Train classical ML baselines
python train_sklearn.py

# 8) Visualize class‑wise spectrograms + differences
python spectrogram_viz.py
```

---

## 2) Configuration (`config.py`)

Global settings for data handling and training live in `config.py`:

- `DATA_CONFIG`: data directory, normalization mode, augmentation toggle/factor, batch size, workers.
- `TRAINING_CONFIG`: epochs, learning rate, weight decay, early‑stop patience, and `model_save_path`.
- `PATHS`: where losses/accuracies are saved.

Update `DATA_CONFIG['data_dir']` to your windowed `.npy` folder before running.

---

## 3) Deep Learning Training (`main.py`)

The primary training script:
- Builds `PSGWindowDataset` for **train** and **val** splits.
- Computes normalization stats on the **training** set and applies them for consistency.
- Creates data loaders, computes **class weights**, and trains a selected model from `models.get_model`.
- Saves the best checkpoint and prints a final evaluation.

Key utilities used: `get_device`, `create_data_loaders`, `print_dataset_info`, `print_sample_info`, `plot_training_curves`, `train_model`, `evaluate_model`, `save_training_history`. It also includes helper functions for PSD plotting/inspection.

**Run:**

```bash
python main.py
```

> Ensure `config.py` points to your data directory and desired hyper‑parameters.

**Notes on normalization:** The script computes training-set normalization stats and applies them to validation to avoid leakage. Verify normalization values during the printed checks.

---

## 4) Data Generation / Labeling Pipeline (`labelling_samples.py`)

A command‑line pipeline to convert **EDF + event TXT files** into windowed `.npy` arrays with labels (`pos`/`neg`), alongside per‑window time arrays and a `channel_names.npy`. It:

- Parses event text files, anchors times to the EDF start timestamp, and computes seconds since start.
- Creates candidate **positive** windows ending `delta` seconds before each hypopnea event and **negative** windows offset by `gamma`, enforcing minimum spacings (`beta`, `min_gap`) and no overlaps.
- Optionally **balances** extra negatives away from forbidden zones around events.
- Extracts slices from EDF at 256 Hz (resampled if needed), saves `(n_channels, n_times)` arrays and corresponding `_times.npy`.
- Writes `channel_names.npy` once.

**CLI Example:**

```bash
python labelling_samples_v2.py   --events-dir /path/to/Events_data   --edf-dir /path/to/EDF_data   --out-dir /path/to/Npy_data   --window-length 10 --delta 5 --beta 20 --gamma 30   --min-gap 60   --channels M1,M2,C3,C4,RIBCAGE,ABDOMEN,SaO2,NASAL PRES.   --max-subjects 50   --preload-edf   -v
```

**Important behaviors & safeguards:**

- EDF is read via MNE; stream is **resampled to 256 Hz** if not already.
- The exporter flags unexpected sample counts (e.g., **20‑second** windows) and logs anomalies.
- It can discover subject IDs present in both Events and EDF directories.
- Windows are persisted as: `<SUBJECT>_win_XXX_<pos|neg>.npy` and `<SUBJECT>_win_XXX_<pos|neg>_times.npy`, plus `channel_names.npy` (once).

---

## 5) Dataset Analysis & QA (`analyze_data.py`)

`EEGDataAnalyzer` provides a convenient **sanity‑check** and **EDA** suite for the saved windows:

- **basic_data_stats**: counts windows, subjects, channels, and label distribution.
- **check_data_integrity**: scans random samples for NaN/Inf, extreme amplitudes, and inconsistent shapes.
- **class_balance_analysis**: saves 4 plots (overall pie, per‑subject pos/neg bars, stacked bars).
- **get_average_samples_per_subject**: quickly estimates sampling density.
- **amplitude_statistics_per_channel**: summary boxplots (mean, median, std, min, max, peak‑to‑peak).
- **sample_visualization**: side‑by‑side random positive vs negative window traces.
- **find_empty_samples**: detects `(4, 0)` empty windows using `*_times.npy` companions.
- **run_full_analysis**: orchestrates the above and prints training recommendations.

Please update the path in the file to point to your dataset path

**Run:**

```bash
python analyze_data.py
```

(Or import `EEGDataAnalyzer` and call methods programmatically.)

---

## 6) Classical ML Baselines (`train_sklearn.py`)

Provides quick **Logistic Regression**, **Random Forest**, and **MLPClassifier** baselines using simple EEG features:

- Applies notch filters (60 Hz and 120 Hz) and computes **Welch PSD** per channel.
- Integrates **band powers** (delta, theta, alpha, beta, gamma) and includes an **alpha/theta ratio** feature.
- Trains/evaluates on features extracted from the same train/val dataset objects used by DL.

**Run:**

```bash
python train_sklearn.py
```

Console will print `classification_report` for each baseline.

---

## 7) Spectrogram Visualization (`spectrogram_viz.py`)

Computes **average spectrograms** (per class) on the training split and visualizes both classes and their **dB difference** with shared color scaling. Functions include:

- `compute_average_spectrogram` — per‑class average over samples/channels.
- `plot_avg_and_diff_spectrograms` — side‑by‑side Class 0, Class 1, and Δ(Class1−Class0) plots with symmetric limits.

**Run:**

```bash
python spectrogram_viz.py
```

---

## 8) Expected Data Layout & Conventions

The generated dataset folder contains:

```
channel_names.npy
SUBJECT_A_win_000_pos.npy
SUBJECT_A_win_000_pos_times.npy
SUBJECT_A_win_001_neg.npy
SUBJECT_A_win_001_neg_times.npy
...
```

- Each window `.npy` holds **(n_channels, n_times)** at **256 Hz** sampling (default 10 s → 2560 samples).
- `*_times.npy` is the aligned time vector per window.
- Labels are encoded in the filename: `pos` or `neg`.

---

## 9) Troubleshooting & Tips

- **No samples found / empty sets** — Check `events_dir`, `edf_dir`, channel names, and CLI filters. Use `-v` for verbose logs.
- **Unexpected window length** — The exporter logs any 20‑second (≈5120 samples) windows; inspect EDF sampling rate and event boundaries for the subject.
- **Normalization drift** — Confirm train‑set stats are applied to validation; inspect printed mean/std on a sample from each split.
- **Class imbalance** — Consider enabling balancing in the generator or using class weights during training.

---

## 10) Dependencies

Create a `requirements.txt` that covers (non‑exhaustive):

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `scikit-learn`, `tqdm`
- `torch` / `torchvision` (as needed by your `models.py`)
- `mne` (EDF handling for the labeling pipeline)

(Adjust versions to your environment/GPU.)

---

## 11) Reproducibility Checklist

- Record `config.py` (learning rate, epochs, batch size, etc.).
- Seed your RNGs (NumPy/PyTorch) in training scripts if deterministic runs are needed.
- Save `train/val` split definition or derive deterministically.
- Archive the generated `.npy` windows with their `_times.npy` and `channel_names.npy` files.

---

### Attributions (file references)

- Training entrypoint and PSD helpers: `main.py`.
- Configuration: `config.py`.
- Data analyzer / QA: `analyze_data.py`.
- Classical baselines: `train_sklearn.py`.
- Spectrograms: `spectrogram_viz.py`.
- EDF→NumPy window exporter: `labelling_samples_v2.py`.

### Checkpoint models
- saved_static_models/best_model - MultiModelEEG with Focal Loss
- saved_static_models/best_model_combined_loss - MultiModelEEG with Combined Loss
- saved_static_models/best_model_lstm_cnn - CNN-LSTM Model with Focal Loss
- saved_static_models/best_model_simple_cnn - CNN Model with Focal Loss
