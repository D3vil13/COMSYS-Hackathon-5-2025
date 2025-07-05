"""
This module provides training and validation functions for task "A" using PyTorch.
Functions:
    train_one_epoch(model, loader, optimizer, device, criterion, task):
        Trains the model for one epoch on the provided data loader.
        Args:
            model (torch.nn.Module): The model to train.
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            device (torch.device): Device to run computations on.
            criterion (callable): Loss function.
            task (str): Task identifier (expects "A").
        Returns:
            dict: Dictionary containing average training loss ('train_loss').
    validate_one_epoch(model, loader, device, criterion, task, gallery_loader=None, gallery_labels=None):
        Validates the model for one epoch on the provided data loader (only for task "A").
        Args:
            model (torch.nn.Module): The model to validate.
            loader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device to run computations on.
            criterion (callable): Loss function.
            task (str): Task identifier (expects "A").
            gallery_loader (torch.utils.data.DataLoader, optional): Not used in this task.
            gallery_labels (list, optional): Not used in this task.
        Returns:
            dict: Dictionary containing computed metrics and average validation loss ('val_loss').
"""
import torch
from metrics import compute_metrics
from config import cfg

def train_one_epoch(model, loader, optimizer, device, criterion, task):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        if task == "A":
            imgs, targets = batch
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return {'train_loss': total_loss / len(loader)}



if cfg.task == "A":
    @torch.no_grad()
    def validate_one_epoch(model, loader, device, criterion, task, gallery_loader=None, gallery_labels=None):
        model.eval()
        all_preds, all_targets = [], []
        total_loss = 0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
        metrics = compute_metrics(all_targets, all_preds)
        metrics['val_loss'] = total_loss / len(loader.dataset)
        torch.cuda.empty_cache()
        return metrics




    



    



