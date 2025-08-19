import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


def get_device():
    """Get the appropriate device for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_data_loaders(train_dataset, val_dataset, batch_size=16, num_workers=0):
    """Create data loaders for training and validation"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def print_dataset_info(train_dataset, val_dataset):
    """Print information about the datasets"""
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Check data distribution in training set
    train_labels = []
    sample_size = min(len(train_dataset), 100)
    
    for i in range(sample_size):
        _, label = train_dataset[i]
        train_labels.append(label.item())
    
    train_class_0 = sum(1 for l in train_labels if l == 0)
    train_class_1 = sum(1 for l in train_labels if l == 1)
    print(f"Training sample check - Class 0: {train_class_0}, Class 1: {train_class_1}")


def print_sample_info(dataset):
    """Print information about a sample from the dataset"""
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    sample_x, sample_y = dataset[0]
    print(f"Sample shape: {sample_x.shape}, Sample label: {sample_y}")
    print(f"Sample data range: [{sample_x.min():.3f}, {sample_x.max():.3f}]")
    print(f"Sample mean: {sample_x.mean():.3f}, std: {sample_x.std():.3f}")


def plot_training_curves(history=None):
    """
    Plot training curves from history or saved files
    
    Args:
        history: Dictionary with training history, or None to load from files
    """
    if history is None:
        try:
            train_loss = np.load("train_losses.npy")
            val_loss = np.load("val_losses.npy")
            train_acc = np.load("train_acc_arr.npy")
            val_acc = np.load("val_acc_arr.npy")
        except FileNotFoundError:
            print("No training history files found!")
            return
    else:
        train_loss = history['train_losses']
        val_loss = history['val_losses']
        train_acc = history['train_accuracies']
        val_acc = history['val_accuracies']
    
    # Create plots
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def get_model_summary(model, input_shape=(1, 4, 512)):
    """Print model summary with parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Try to get model output shape
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            output = model(dummy_input)
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Could not determine output shape: {e}")


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint with additional information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_class': model.__class__.__name__
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")
        
        return epoch, loss
    except FileNotFoundError:
        print(f"Checkpoint file {filepath} not found!")
        return 0, 0.0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0.0