"""
Logger module for integrating with Weights & Biases (wandb) for experiment tracking.
Functions:
    setup_wandb():
        Initializes a wandb run if enabled in the configuration.
    log_metrics(metrics, step):
        Logs a dictionary of metrics to wandb at a specific step if enabled.
    finish_wandb():
        Finishes the current wandb run if enabled.
"""

from config import cfg
import wandb

def setup_wandb():
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

def log_metrics(metrics, step):
    if cfg.use_wandb:
        wandb.log(metrics, step=step)

def finish_wandb():
    if cfg.use_wandb:
        wandb.finish()
