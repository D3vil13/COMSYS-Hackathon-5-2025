"""
data_loader.py
This module provides data loading utilities for two tasks, "A" and "B", as specified in the configuration (cfg.task):
Task "A":
    - Uses torchvision.datasets.ImageFolder to load training and validation datasets.
    - Provides a `get_dataloaders` function that returns PyTorch DataLoaders for training and validation, the number of classes, and the class-to-index mapping.
Task "B":
    - Defines custom PyTorch Dataset classes for triplet loss training and evaluation:
        - TripletFaceDataset: Generates triplets (anchor, positive, negative) for face recognition tasks, including support for distorted images.
        - ReferenceDataset: Loads reference (undistorted) images for each identity.
        - DistortedQueryDataset: Loads distorted images for each identity.
    - Provides a `get_dataloaders` function that returns DataLoaders for triplet training, reference validation, and distorted query validation.
Dependencies:
    - os, random, glob, PIL.Image, torch, torchvision.datasets, torch.utils.data.Dataset/DataLoader
    - src.config.cfg for configuration parameters (task, directories, batch size, etc.)
Functions and Classes:
    - get_dataloaders(train_transforms, val_transforms): Returns appropriate DataLoaders depending on the task.
    - TripletFaceDataset: Custom Dataset for generating triplets for metric learning.
    - ReferenceDataset: Custom Dataset for loading reference images.
    - DistortedQueryDataset: Custom Dataset for loading distorted query images.
Usage:
    Import and call `get_dataloaders` with appropriate transforms to obtain DataLoaders for training and evaluation.
"""

import os
import random
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from config import cfg
import torch
from glob import glob

#
if cfg.task == "A":
    def get_dataloaders(train_transforms, val_transforms):
        train_dataset = datasets.ImageFolder(cfg.train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(cfg.val_dir, transform=val_transforms)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        num_classes = len(train_dataset.classes)
        return train_loader, val_loader, num_classes, train_dataset.class_to_idx



#
elif cfg.task == "B":
    class TripletFaceDataset(Dataset):
        def __init__(self, root_dir, transform):
            self.root_dir = root_dir
            self.transform = transform
            self.triplets = self._create_triplets()

        def _create_triplets(self):
            triplets = []
            persons = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

            for person in persons:
                person_dir = os.path.join(self.root_dir, person)
                pos_imgs = glob(os.path.join(person_dir, '*.jpg'))
                pos_imgs += glob(os.path.join(person_dir, '*.png'))

                dist_dir = os.path.join(person_dir, 'distortion')
                if os.path.exists(dist_dir):
                    pos_imgs += glob(os.path.join(dist_dir, '*.jpg'))
                    pos_imgs += glob(os.path.join(dist_dir, '*.png'))

                if len(pos_imgs) < 2:
                    continue

                for _ in range(5):  # Try creating 5 triplets per identity
                    anchor = random.choice(pos_imgs)
                    positive = random.choice([p for p in pos_imgs if p != anchor])

                    # Instead of 1 negative, try more and sample one
                    negative = None
                    tries = 0
                    while tries < 10:
                        negative_person = random.choice([p for p in persons if p != person])
                        neg_dir = os.path.join(self.root_dir, negative_person)
                        neg_imgs = glob(os.path.join(neg_dir, '*.jpg')) + glob(os.path.join(neg_dir, '*.png'))
                        neg_dist = os.path.join(neg_dir, 'distortion')
                        if os.path.exists(neg_dist):
                            neg_imgs += glob(os.path.join(neg_dist, '*.jpg')) + glob(os.path.join(neg_dist, '*.png'))
                        if neg_imgs:
                            negative = random.choice(neg_imgs)
                            break
                        tries += 1

                    if negative:
                        triplets.append((anchor, positive, negative))
                neg_dir = os.path.join(self.root_dir, negative_person)
                neg_imgs = glob(os.path.join(neg_dir, '*.jpg')) + glob(os.path.join(neg_dir, '*.png'))
                neg_dist = os.path.join(neg_dir, 'distortion')
                if os.path.exists(neg_dist):
                    neg_imgs += glob(os.path.join(neg_dist, '*.jpg')) + glob(os.path.join(neg_dist, '*.png'))

                if not neg_imgs:
                    continue
                negative = random.choice(neg_imgs)

                triplets.append((anchor, positive, negative))

            return triplets

        def __getitem__(self, idx):
            anc_path, pos_path, neg_path = self.triplets[idx]
            anchor = self.transform(Image.open(anc_path).convert('RGB'))
            positive = self.transform(Image.open(pos_path).convert('RGB'))
            negative = self.transform(Image.open(neg_path).convert('RGB'))
            return anchor, positive, negative

        def __len__(self):
            return len(self.triplets)


    class ReferenceDataset(Dataset):
        def __init__(self, root_dir, transform):
            self.samples = []
            self.label_map = {}
            self.transform = transform

            for label, person in enumerate(sorted(os.listdir(root_dir))):
                person_dir = os.path.join(root_dir, person)
                if not os.path.isdir(person_dir):
                    continue
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith(('jpg', 'png')):
                        self.samples.append((os.path.join(person_dir, img_name), label))
                        self.label_map[label] = person

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            return self.transform(image), label


    class DistortedQueryDataset(Dataset):
        def __init__(self, root_dir, transform):
            self.samples = []
            self.label_map = {}
            self.transform = transform

            for label, person in enumerate(sorted(os.listdir(root_dir))):
                dist_dir = os.path.join(root_dir, person, 'distortion')
                if not os.path.exists(dist_dir):
                    continue
                for img_name in os.listdir(dist_dir):
                    if img_name.lower().endswith(('jpg', 'png')):
                        self.samples.append((os.path.join(dist_dir, img_name), label))
                        self.label_map[label] = person

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            return self.transform(image), label


    def get_dataloaders(train_transform, val_transform):
        if cfg.task == "A":
            raise NotImplementedError("Task A uses a different loader")
        elif cfg.task == "B":
            train_dataset = TripletFaceDataset(cfg.train_dir, train_transform)
            val_ref_dataset = ReferenceDataset(cfg.val_dir, val_transform)
            val_query_dataset = DistortedQueryDataset(cfg.val_dir, val_transform)

            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
            val_ref_loader = DataLoader(val_ref_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
            val_query_loader = DataLoader(val_query_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

            return train_loader, val_ref_loader, val_query_loader
