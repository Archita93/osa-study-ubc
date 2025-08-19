"""
Configuration file for PSG classification training
"""

# Data configuration
DATA_CONFIG = {
    'data_dir': 'path/to/your/data_directory',  # Update with your actual data directory
    'normalize': "robust",
    'use_augmentation': False,  # Start without augmentation for debugging
    'augment_factor': 1,
    'batch_size': 32,
    'num_workers': 16
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 200,
    'model_save_path': './saved_static_models/best_model_simple_cnn.pth'
}


# Paths for saving results
PATHS = {
    'model_checkpoint': './saved_static_models/best_model_simple_cnn.pth',
    'train_losses': './saved_static_models/train_losses.npy',
    'val_losses': './saved_static_models/val_losses.npy',
    'train_accuracies': './saved_static_models/train_acc_arr.npy',
    'val_accuracies': './saved_static_models/val_acc_arr.npy'
}
