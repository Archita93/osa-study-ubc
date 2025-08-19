import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
    
class CombinedLoss(nn.Module):
    """Combined Focal Loss + Label Smoothing Loss"""
    def __init__(self, n_classes=2, focal_alpha=1, focal_gamma=2, 
                 smoothing=0.1, focal_weight=0.7, smooth_weight=0.3):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.smooth_loss = LabelSmoothingLoss(classes=n_classes, smoothing=smoothing)
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        smooth = self.smooth_loss(pred, target)
        return self.focal_weight * focal + self.smooth_weight * smooth