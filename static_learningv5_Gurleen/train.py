import numpy as np
import torch
from tqdm import tqdm
from loss import CombinedLoss


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc='Training')):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        try:
            logits = model(X)
            loss = criterion(logits, y)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                continue
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    
    return avg_train_loss, train_acc


def validate_epoch(model, val_loader, criterion, device, scheduler):
    """Validate the model for one epoch"""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc='Validation'):
            X, y = X.to(device), y.to(device)
            
            try:
                logits = model(X)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                
            except Exception as e:
                print(f"Error in validation: {e}")
                continue
    scheduler.step()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    return avg_val_loss, val_acc


def train_model(model, train_loader, val_loader, config, device):
    """
    Complete training loop
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary with training configuration
        device: Device to run training on
    
    Returns:
        Dictionary with training history
    """
    # Setup optimizer and criterion
    criterion = CombinedLoss(n_classes=2, focal_weight=0.7, smooth_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {config['epochs']} epochs...")
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, scheduler)
        
        # Save history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), config.get('model_save_path', 'best_model.pth'))
            print(f"  → New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config.get('patience', 10):
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save training curves
        save_training_history(history)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history


def evaluate_model(model, val_loader, device):
    
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    """Evaluate model performance with detailed metrics"""
    model.eval()
    
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        class_correct = [0, 0]
        class_total = [0, 0]
        y_t, y_p = [], []
        
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=1)
            
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
            
            # Per-class accuracy
            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if pred[i] == y[i]:
                    class_correct[label] += 1
            # Calculate F1, precision, recall
            y_true = y.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_t.append(y_true)
            y_p.append(y_pred)
        flat_list_pred = [
                            x
                            for xs in y_p
                            for x in xs
                        ]
        
        flat_list = [
                        x
                        for xs in y_t
                        for x in xs
                    ]
        
        f1 = f1_score(flat_list, flat_list_pred)
        precision = precision_score(flat_list, flat_list_pred)
        recall = recall_score(flat_list, flat_list_pred)
        cm = confusion_matrix(flat_list, flat_list_pred)
        print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        results = {
            'overall_accuracy': total_correct / total_samples,
            'class_0_accuracy': class_correct[0] / max(class_total[0], 1),
            'class_1_accuracy': class_correct[1] / max(class_total[1], 1),
            'class_0_samples': class_total[0],
            'class_1_samples': class_total[1]
        }
        
        print(f"\nFinal Results:")
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Class 0 Accuracy: {results['class_0_accuracy']:.4f} ({results['class_0_samples']} samples)")
        print(f"Class 1 Accuracy: {results['class_1_accuracy']:.4f} ({results['class_1_samples']} samples)")

        # add f1 score, precision, recall, and confusion matrix
        
        return results


def save_training_history(history):
    """Save training history to numpy files"""
    np.save('train_losses.npy', np.array(history['train_losses']))
    np.save('val_losses.npy', np.array(history['val_losses']))
    np.save('train_acc_arr.npy', np.array(history['train_accuracies']))
    np.save('val_acc_arr.npy', np.array(history['val_accuracies']))


def load_training_history():
    """Load training history from numpy files"""
    try:
        history = {
            'train_losses': np.load("train_losses.npy"),
            'val_losses': np.load("val_losses.npy"), 
            'train_accuracies': np.load("train_acc_arr.npy"),
            'val_accuracies': np.load("val_acc_arr.npy")
        }
        return history
    except FileNotFoundError as e:
        print(f"Training history file not found: {e}")
        return None