import torch
import torch.nn as nn


class FocalLossCE(torch.nn.Module):
    def __init__(self, weight=None, alpha=1.0, gamma=2.0, label_smoothing=0.0):
        super(FocalLossCE, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, target):
        ce_loss = self.ce(preds, target.long())
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        f_loss = torch.mean(f_loss)
        return f_loss


class FocalLossBCE(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
