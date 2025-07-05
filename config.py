"""
Config class for managing settings and hyperparameters for face-related tasks.
Attributes:
    task (str): Task selection. "A" for Gender Classification, "B" for Face Matching.
    train_dir (str): Path to the training data directory, based on selected task.
    val_dir (str): Path to the validation data directory, based on selected task.
    num_classes (int or None): Number of output classes. 2 for Gender Classification, None for Face Matching.
    backbone (str): Model backbone architecture. "resnet101" for Task A, "efficientnet_b3" for Task B.
    pretrained (bool): Whether to use pretrained weights for the model backbone.
    embedding_dim (int): Dimension of the embedding layer.
    dropout_rate (float): Dropout rate used in the model.
    triplet_margin (float): Margin value for triplet loss (used in face matching).
    threshold (float): Threshold for decision making (e.g., similarity threshold in face matching).
    batch_size (int): Number of samples per batch during training.
    num_workers (int): Number of worker threads for data loading.
    epochs (int): Number of training epochs.
    lr (float): Learning rate for the optimizer.
    weight_decay (float): Weight decay (L2 regularization) for the optimizer.
    patience (int): Early stopping patience (number of epochs with no improvement).
    input_size (int): Input image size (height and width).
    use_blur (bool): Whether to apply blur augmentation.
    use_brightness (bool): Whether to apply brightness augmentation.
    use_fog (bool): Whether to apply fog augmentation.
    use_greyscale (bool): Whether to apply greyscale augmentation.
    use_rain (bool): Whether to apply rain augmentation.
    use_wandb (bool): Whether to use Weights & Biases for experiment tracking.
    wandb_project (str): Name of the Weights & Biases project.
    seed (int): Random seed for reproducibility.
Usage:
    Instantiate the configuration with `cfg = Config()`.
    Access configuration parameters as attributes of `cfg`.
"""

import os
import sys
class Config:
    # Task selection
    script_name = os.path.basename(sys.argv[0]).lower()

    if "task_a" in script_name or "inference_task_a" in script_name:
        task = "A"
    elif "task_b" in script_name or "inference_task_b" in script_name:
        task = "B"
    else:
        # Default to B if run_training.py is executed directly
        task = "B"

    # Paths (update these to your environment)
    if task == "A":
        train_dir = "D:\\Comys_Hackathon5\\Comys_Hackathon5\\Task_A\\train"
        val_dir = "D:\\Comys_Hackathon5\\Comys_Hackathon5\\Task_A\\val"
        test_dir = "enter the test directory path here"
        num_classes = 2
        backbone = "resnet101"
        embedding_dim = 128
    elif task == "B":
        train_dir = "D:\\Comys_Hackathon5\\Comys_Hackathon5\\Task_B\\train"
        val_dir = "D:\\Comys_Hackathon5\\Comys_Hackathon5\\Task_B\\val"
        test_dir = "enter the test directory path here"
        
        num_classes = None  # To be inferred from folders
        backbone = "efficientnet_b3"
        embedding_dim = 512
        

    # Model settings
    
    pretrained = True
 
    dropout_rate = 0.4
    triplet_margin = 0.15  # Slightly increased
    threshold = 0.8  # Will be optimized automatically

    # Training params
    batch_size = 20
    num_workers = 8
    epochs = 15
    lr = 5e-5  # Lower learning rate for EfficientNet
    weight_decay = 1e-4
    patience = 5  # Early stopping

    # Input preprocessing
    input_size = 224
    use_blur = True
    use_brightness = True
    use_fog = True
    use_greyscale = True
    use_rain = True

    # Logging
    use_wandb = False
    wandb_project = "facecom-pipeline"

    # Reproducibility
    seed = 42

cfg = Config()
