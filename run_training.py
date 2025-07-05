"""
Main training script for configurable deep learning tasks (Task A: classification, Task B: metric learning).
This script sets up data loaders, model, loss, optimizer, and scheduler based on the selected task in the configuration.
It supports:
- Task A: Standard classification with class imbalance handling.
- Task B: Metric learning with improved embedding model, triplet loss, and advanced training strategies.
Features:
- Device selection (CUDA/CPU).
- Data augmentation and transformation.
- Class weight computation for imbalanced datasets.
- Model selection and initialization based on task.
- Optimizer and scheduler setup (Adam/AdamW, StepLR/CosineAnnealingLR).
- Training and validation loops with metric logging.
- Early stopping based on F1 score.
- Model checkpointing (with additional info for Task B).
- Integration with Weights & Biases for experiment tracking.
Functions:
    main(): Orchestrates the full training and validation process according to the configuration.
Usage:
    Run as a script to start training with the specified configuration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from config import cfg
from data_loader import get_dataloaders
from augmentations import get_train_transforms, get_val_transforms, AlbumentationsTransform

from logger import setup_wandb, log_metrics, finish_wandb
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def main():
    setup_wandb()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    train_transforms = AlbumentationsTransform(get_train_transforms())
    val_transforms = AlbumentationsTransform(get_val_transforms())

    # Loaders
    if cfg.task == "A":
        from A_train import train_one_epoch, validate_one_epoch  
        train_loader, val_loader, num_classes, class_to_idx = get_dataloaders(train_transforms, val_transforms)
        cfg.num_classes = num_classes
        # Extract labels from dataset
        train_labels = [sample[1] for sample in train_loader.dataset.samples]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    elif cfg.task == "B":
        train_loader, val_ref_loader, val_query_loader = get_dataloaders(train_transforms, val_transforms)

    # Model
    if cfg.task == "A":
        from A_model import SimpleClassifier
        model = SimpleClassifier(cfg.backbone, cfg.num_classes, cfg.task, pretrained=cfg.pretrained).to(device)
        # Class imbalance weights (as in your Task A)
        weights = torch.tensor([1.0 / 303, 1.0 / 1623], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        
    
    elif cfg.task == "B":
        # Import improved model and training components
        from B_model import ImprovedEmbeddingModel, ImprovedTripletLoss
        from B_train import ImprovedTrainer
        
        # Use improved model with better regularization
        model = ImprovedEmbeddingModel(
            backbone=cfg.backbone, 
            embedding_dim=cfg.embedding_dim, 
            pretrained=cfg.pretrained,
            dropout_rate=getattr(cfg, 'dropout_rate', 0.3)
        ).to(device)
        
        

        
        # Use improved triplet loss with hard mining
        criterion = ImprovedTripletLoss(margin=cfg.triplet_margin, hard_mining=True)
        
        # Use AdamW optimizer with better settings
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        # Use cosine annealing scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        
        # Initialize improved trainer
        trainer = ImprovedTrainer(model, device, threshold=cfg.threshold, enable_plots=False)


    best_f1 = 0.0
    patience = getattr(cfg, 'patience', 5)
    patience_counter = 0
    
    for epoch in range(cfg.epochs):
        if cfg.task == "A":
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, criterion, cfg.task)
            val_metrics = validate_one_epoch(model, val_loader, device, criterion=criterion, task=cfg.task)
            
        elif cfg.task == "B":
            # Use improved training methods
            train_metrics = trainer.train_one_epoch(train_loader, optimizer, criterion, epoch)
            val_metrics = trainer.validate_comprehensive(val_query_loader, val_ref_loader, epoch)
            
            # Update threshold based on optimal value found during validation
            if 'optimal_threshold' in val_metrics:
                trainer.threshold = val_metrics['optimal_threshold']

        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        if cfg.task == "B":
            print(f"ROC AUC: {val_metrics['roc_auc']:.4f}, Optimal Threshold: {val_metrics.get('optimal_threshold', cfg.threshold):.4f}")

        log_metrics({**train_metrics, **val_metrics}, step=epoch)
        scheduler.step()

        # Enhanced early stopping for Task B
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{cfg.task}_model.pth")
            
            # Save additional info for Task B
            if cfg.task == "B":
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1': best_f1,
                    'optimal_threshold': val_metrics.get('optimal_threshold', cfg.threshold),
                    'config': cfg.__dict__
                }, f"best_{cfg.task}_model_complete.pth")
                
        else:
            patience_counter += 1
            if cfg.task == "B" and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in F1 score")
                break

    print(f"Training completed. Best F1: {best_f1:.4f}")
    finish_wandb()


if __name__ == "__main__":
    main()