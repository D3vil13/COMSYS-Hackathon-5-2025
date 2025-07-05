"""
Defines a TripletLoss class for deep metric learning.
Classes:
    TripletLoss(nn.Module):
        Computes the triplet margin loss between anchor, positive, and negative samples.
        Args:
            margin (float, optional): Margin for the triplet loss. Default is 0.2.
        Methods:
            forward(anchor, positive, negative):
                Calculates the triplet margin loss given anchor, positive, and negative embeddings.
                Args:
                    anchor (Tensor): Anchor embeddings of shape (N, D).
                    positive (Tensor): Positive embeddings of shape (N, D).
                    negative (Tensor): Negative embeddings of shape (N, D).
                Returns:
                    Tensor: Computed triplet loss value.
"""
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)
