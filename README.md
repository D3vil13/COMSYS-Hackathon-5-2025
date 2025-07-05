# 🧠 FaceCom-Pipeline: Gender Classification & Face Matching

This project contains two deep learning tasks implemented using **PyTorch**:

- **Task A**: *Gender Classification* (Binary Classification using ResNet)
- **Task B**: *Face Matching* (Metric Learning using Triplet Loss with EfficientNet)

All training, validation, and inference workflows are **modular** and ready-to-run.

---

## 🗂 Directory Structure

```
project_root/
├── A_model.py                  # Task A: ResNet-based gender classifier
├── A_train.py                  # Task A: training loop
├── B_model.py                  # Task B: EfficientNet + embedding + loss
├── B_train.py                  # Task B: improved training with Triplet Loss and metrics
├── augmentations.py            # Transform pipelines using Albumentations
├── config.py                   # Global configuration (task, paths, hyperparameters)
├── data_loader.py              # Dataset classes for both tasks
├── logger.py                   # Optional logging module
├── metrics.py                  # Metrics like accuracy, F1, AUC, etc.
├── losses.py                   # Triplet and ArcFace losses
├── run_training.py             # Entry point for training
├── task_a.py                   # Sets cfg.task="A", set train_dir and val_dir in config.py and runs training
├── task_b.py                   # Sets cfg.task="B", set train_dir and val_dir in config.py and runs training
├── inference_task_a.py         # Loads saved model for Task A and performs inference (set test directory in this file only)
├── inference_task_b.py         # Loads saved model for Task B and performs matching (set test directory in this file only)
├── best_A_model.pth            # Saved best models for task A (optional)
├── best_B_model_complete.pth   # Saved best models for task B (optional)
└── README.md                   # You're here
```

## 🔍 How to Run Inference

### 🎯 Task A: Gender Classification

**Test Folder Format:**
```
test/
  ├── male/
  └── female/
```
**Set test directory in inferance_task_a.py file:**

**Run Inference:**
```bash
python inference_task_a.py
```

- Outputs predictions and confidence.


---

### 🎯 Task B: Face Matching

**Test Folder Format:**
```
test/
  ├── person1/
  │     └── distortion/
  │           └── distorted1.jpg
  └── person2/
        └── distortion/
              └── distorted2.jpg
```

**Set test directory in inferance_task_b.py file:**

**Run Inference:**
```bash
python inference_task_b.py
```

- Matches each distorted image to the closest identity in the validation set.
- Outputs predicted identities and cosine similarities.

---

## 🏋️‍♂️ How to Train

### ✅ Task A: Gender Classification

**Input Folder Format** (ImageFolder):
```
train/
  ├── male/
  └── female/
val/
  ├── male/
  └── female/
```
Sets cfg.task="A", set train_dir and val_dir in config.py and runs training
**Command to train:**
```bash
python task_a.py
```

---

### ✅ Task B: Face Matching (Triplet Loss)

**Input Folder Format:**
```
train/
  └── person1/
        ├── img1.jpg
        └── distortion/
              └── distorted1.jpg

val/
  └── person1/
        ├── img2.jpg
        └── distortion/
              └── distorted2.jpg
```
Sets cfg.task="B", set train_dir and val_dir in config.py and runs training
**Command to train:**
```bash
python task_b.py
```

---


---

## 🧬 Model Architectures

### 🟩 Task A: Gender Classification

- **Backbone**: ResNet101 *(configurable)*
- **Head**: Dropout → Linear (2 classes)
- **Loss**: Weighted CrossEntropyLoss
- **Pretrained**: Yes *(default)*
- **Augmentations**:
  - Blur
  - Brightness
  - Fog
  - Rain
  - Greyscale

---

### 🟦 Task B: Face Matching

- **Backbone**: EfficientNet-B3 *(configurable)*
- **Head**: Dropout → Linear → BatchNorm → L2 Normalized
- **Loss**: Improved Triplet Loss + L2 Regularization
- **Validation**: Multi-reference cosine similarity
- **Metrics**:
  - F1 Score
  - Precision / Recall
  - ROC AUC
- **Plots (optional)**:
  - Similarity Distribution
  - t-SNE
  - Confusion Matrix

---

## 💾 Model Weights

Weights are saved automatically after training:
```
weights/best_A_model.pth
weights/best_B_model.pth
```

You can set your preferred path in the training loop or move weights to the `weights/` directory.

---

## 📝 Notes

- ✅ Update only the **paths** in `config.py` before running.
- 🖼 Disable validation plots in Task B by setting:
  ```python
  enable_plots = False
  ```
- 🧪 Both tasks support **test-time evaluation** using saved weights.
- 🖥 Compatible with **Windows and Linux** (ensure correct path formatting).

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/D3vil13/facecom-pipeline.git
cd facecom-pipeline

pip install -r requirements.txt


# For Gender Classification (Task A)
python task_a.py

# For Face Matching (Task B)
python task_b.py
```

Made with ❤️ for research and learning.