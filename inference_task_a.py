

import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from A_model import SimpleClassifier
from augmentations import AlbumentationsTransform, get_val_transforms
from config import cfg
import torch.nn.functional as F
from sklearn.metrics import classification_report

# Set Task A
cfg.task = "A"
test_dir = "enter test directory here"  # Ensure test_dir is set correctly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(num_classes):
    model = SimpleClassifier(cfg.backbone, num_classes, cfg.task, pretrained=False).to(device)
    model.load_state_dict(torch.load("best_A_model.pth", map_location=device))
    model.eval()
    return model

def infer_on_directory(test_dir):
    transform = AlbumentationsTransform(get_val_transforms())
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    cfg.num_classes = len(dataset.classes)
    model = load_model(cfg.num_classes)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

if __name__ == "__main__":
    infer_on_directory(test_dir)
