"""
This module provides an improved deep metric learning model with advanced loss functions and data augmentation strategies.
Classes:
    ImprovedEmbeddingModel(nn.Module):
        A flexible embedding model supporting EfficientNet-B3, ResNet50, and ResNet101 backbones.
        Features an enhanced embedding head with dropout and batch normalization, and outputs L2-normalized embeddings.
    ArcMarginProduct(nn.Module):
        Implements the ArcFace margin-based softmax layer for improved classification with angular margin penalty.
    ImprovedTripletLoss(nn.Module):
        Computes the triplet loss with optional online hard mining for better separation of positive and negative pairs.
    CenterLoss(nn.Module):
        Implements Center Loss to encourage intra-class compactness and inter-class separability in the embedding space.
Functions:
    get_enhanced_transforms():
        Returns:
            train_transform (torchvision.transforms.Compose): Data augmentation pipeline for training.
            val_transform (torchvision.transforms.Compose): Preprocessing pipeline for validation.
        Provides advanced data augmentation including resizing, flipping, rotation, color jitter, affine transforms, grayscale, normalization, and noise injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import math

class ImprovedEmbeddingModel(nn.Module):
    def __init__(self, backbone="efficientnet_b3", embedding_dim=128, pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        if backbone == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b3(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
            
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Enhanced embedding head with regularization
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get embeddings
        embeddings = self.embedding_head(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition"""
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > math.cos(math.pi - self.margin), phi, cosine - math.sin(math.pi - self.margin) * self.margin)
        
        # Convert label to one hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

class ImprovedTripletLoss(nn.Module):
    """Improved Triplet Loss with online hard mining"""
    def __init__(self, margin=0.3, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        
    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Basic triplet loss
        basic_loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.hard_mining:
            # Hard negative mining - focus on hardest triplets
            hard_triplets = basic_loss > 0
            if hard_triplets.sum() > 0:
                return basic_loss[hard_triplets].mean()
            else:
                return basic_loss.mean()
        else:
            return basic_loss.mean()

class CenterLoss(nn.Module):
    """Center Loss for better clustering"""
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

# Data augmentation strategies for better generalization
import torchvision.transforms as transforms

def get_enhanced_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Add noise augmentation
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform