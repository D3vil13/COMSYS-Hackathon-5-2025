"""
This module defines a simple image classifier model using a configurable ResNet backbone.
Classes:
    SimpleClassifier: A neural network module that uses a ResNet backbone (e.g., ResNet101) for feature extraction,
    followed by a linear classifier for a specified number of output classes. The backbone can be initialized with
    pretrained weights. The classifier is suitable for image classification tasks.
Attributes:
    what_weights: Reference to torchvision's ResNet101_Weights for selecting pretrained weights.
Usage:
    Instantiate SimpleClassifier with the desired backbone name (e.g., 'resnet101'), number of classes, task identifier,
    and whether to use pretrained weights.
Example:
    model = SimpleClassifier('resnet101', num_classes, task='A', pretrained=True)
"""
import torch.nn as nn
import torchvision.models as models
from config import cfg

from torchvision.models import ResNet101_Weights

what_weights = ResNet101_Weights


if cfg.task == "A":
    class SimpleClassifier(nn.Module):
        def __init__(self, backbone_name, num_classes, task, pretrained=True):
            super().__init__()
            weights = what_weights.DEFAULT if pretrained else None
            backbone = getattr(models, backbone_name)(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.task = task
            self.classifier = nn.Linear(in_features, num_classes)
            
        def forward(self, x, labels=None):
            features = self.backbone(x)
            return self.classifier(features)

