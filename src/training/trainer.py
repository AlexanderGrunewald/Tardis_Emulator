"""
TARDIS Emulator - Trainer module

This module provides functionality for training PyTorch models for the TARDIS emulator.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
import logging
import datetime
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from  torch.utils.data import DataLoader

import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_util import TrainingConfig
from src.utils.data_manager import DataManager
from src.models.tarids_emulator import Emulator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0, best_model_path = None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_path = best_model_path
        self.save_best = best_model_path is not None

    def step(self, loss: float, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if self.save_best and model is not None:
                self._save_model(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False
    def _save_model(self, model):
        if self.best_model_path:
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(model.state_dict(), self.best_model_path)
            logger.info(f"Saved best model to {self.best_model_path}")


def create_prediction_plot(true_values, pred_values, title):

    # Ensure tensors are on CPU and detached from computation graph
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().detach()
    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.cpu().detach()

    # Convert tensors to numpy arrays
    true_np = true_values.numpy()
    pred_np = pred_values.numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the number of variables
    n_vars = true_np.shape[0]
    x = np.arange(n_vars)

    # Plot bars for true and predicted values
    ax.bar(x - 0.2, true_np, width=0.4, label='True Values', color='blue', alpha=0.7)
    ax.bar(x + 0.2, pred_np, width=0.4, label='Predictions', color='red', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tight layout
    fig.tight_layout()

    return fig


class Trainer:
    def __init__(self,
                 train_config_path: str,
                 model: nn.Module = None,
                 loss_fn: Optional[Callable] = None,
                 metrics: Optional[Dict[str, Callable]] = None):
        try:
            self.train_config = TrainingConfig(train_config_path)
        except FileNotFoundError:
            logger.error(f"No training config file found at {train_config_path}")
            raise
        except Exception as e:
            logger.error(f"Error Loading Configuration {e}")
            raise ValueError(f"Invalid configuration{e}")
        try:
            self.data_manager = DataManager(self.train_config)
        except Exception as e:
            logger.error(f"Error Loading DataManager {e}")
            raise
        self.device = self._set_up_device()

        self.model = self._setup_model(model)

        self.loss_fn = loss_fn if loss_fn is not None else F.mse_loss
        self.metrics = metrics if metrics is not None else defaultdict()

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.checkpoint_dir = self._setup_checkpoint_dir()

        # Flag to track whether wandb was successfully initialized
        self.wandb_initialized = False
        self._setup_wandb()


    def _set_up_device(self):
        if torch.cuda.is_available():
            logger.info("Using CUDA device.")
            return torch.device("cuda")
        else:
            logger.info("Using CPU device.")
            return torch.device("cpu")

    def _setup_model(self, model: Optional[nn.Module]) -> nn.Module:

        if model is not None:
            logger.info(f"Using provided model: {type(model).__name__}")
            model.to(self.device)
            return model
        try:
            # Get model config path from training config
            model_config_path = self.train_config.resolve_path(
                self.train_config.get_nested_value("model_config")
            )
            logger.info(f"Creating model from config: {model_config_path}")
            model = Emulator(model_config_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise ValueError(f"Could not create model: {e}")

    def _create_optimizer(self):
        train_params = self.train_config.get_training_params()

        lr = train_params.get("learning_rate", 0.001)
        opt_name = train_params.get("optimizer", "adam").lower()
        wd = train_params.get("weight_decay", 0.0)

        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise NotImplementedError(f"Optimizer {opt_name} not implemented.")

    def _create_scheduler(self):

        train_params = self.train_config.get_training_params()
        scheduler_name = train_params.get("scheduler", "none").lower()
        scheduler_params = train_params.get("scheduler_params", {})

        if scheduler_name == "none" or not scheduler_name:
            logger.info("No scheduler configured")
            return None

        logger.info(f"Creating scheduler: {scheduler_name}")

        if scheduler_name == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=scheduler_params.get("patience", 10),
                factor=scheduler_params.get("factor", 0.5),
                min_lr=scheduler_params.get("min_lr", 1e-4),
            )

        elif scheduler_name == "steplr":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get("step_size", 10),
                gamma=scheduler_params.get("gamma", 0.1)
            )
        elif scheduler_name == "cosineannealinglr":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get("t_max", 10),
                eta_min=scheduler_params.get("eta_min", 0))

        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using no scheduler")
            return None

    def _setup_checkpoint_dir(self) -> str:

        logging_params = self.train_config.get_logging_params()
        checkpoint_dir = logging_params.get("checkpoint_dir", "results/checkpoints")

        checkpoint_dir = self.train_config.resolve_path(checkpoint_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir

    def _setup_data(self):
        try:
            data_loaders = self.data_manager.prepare_data()
            return data_loaders
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise RuntimeError(f"Data preparation failed: {e}")

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        Save a checkpoint of the model.

        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
        """
        # Get checkpoint parameters from config
        logging_params = self.train_config.get_logging_params()
        save_frequency = logging_params.get("save_frequency", 10)
        save_best_only = logging_params.get("save_best_only", True)

        # Skip if not at save frequency and not saving best only
        if save_best_only or epoch % save_frequency != 0:
            return

        # Create checkpoint filename
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for the current batch.

        Args:
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): Ground truth targets

        Returns:
            Dict[str, float]: Dictionary of metric values
        """
        results = {}

        for name, metric_fn in self.metrics.items():
            try:
                results[name] = metric_fn(outputs, targets).item()
            except Exception as e:
                logger.warning(f"Error computing metric {name}: {e}")
                results[name] = float('nan')

        return results

    def _setup_wandb(self) -> None:
        """
        Set up Weights & Biases for experiment tracking if enabled.
        """
        # Get wandb parameters from config
        logging_params = self.train_config.get_logging_params()
        use_wandb = logging_params.get("wandb", False)

        if not use_wandb:
            logger.info("Weights & Biases logging is disabled in config")
            return

        project_name = logging_params.get("wandb_project", "tardis-emulator")
        entity_name = logging_params.get("wandb_entity", "tardis-emulator")
        run_name = logging_params.get("wandb_run_name", f"run-{datetime.time().strftime('%Y%m%d-%H%M%S')}")

        # Initialize wandb
        try:
            wandb.init(entity=entity_name, project=project_name, name=run_name, config=self.train_config.get_config())
            logger.info(f"Initialized Weights & Biases: project={project_name}, run={run_name}")
            self.wandb_initialized = True
        except Exception as e:
            logger.error(f"Error initializing Weights & Biases: {e}")
            logger.warning("Weights & Biases logging will be disabled")

    def train(self):

        try:
            data_loaders = self._setup_data()
            train_loader = data_loaders["train"]
            test_loader = data_loaders["test"]
            val_loader = data_loaders["val"]
        except Exception as e:
            logger.error(f"Error setting up data loaders: {e}")
            raise RuntimeError(f"Failed to set up data loaders: {e}")

        train_params = self.train_config.get_training_params()
        n_epochs = train_params.get("num_epochs", 100)


        early_stop_params = train_params.get("early_stopping", {})
        early_stop = None

        if early_stop_params.get("enabled", False):
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            early_stop = EarlyStopping(
                patience=early_stop_params.get("patience", 10),
                min_delta=early_stop_params.get("min_delta", 0.0),
                best_model_path=best_model_path
            )
            logger.info(f"Early stopping enabled with patience={early_stop.patience}")

        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

        for metric_name in self.metrics.keys():
            history[f"train_{metric_name}"] = []
            history[f"val_{metric_name}"] = []

        logger.info(f"Starting training for {n_epochs} epochs")
        start_time = time.time()
        try:
            for i, epoch in enumerate(range(n_epochs)):
                epoch_start_time = time.time()

                self.model.train()
                train_loss = 0
                train_metrics = {name: 0.0 for name in self.metrics.keys()}
                train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")

                for x_batch, y_batch in train_iter:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(x_batch)
                    loss = F.mse_loss(output, y_batch)

                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    if self.metrics:
                        batch_metrics = self._compute_metrics(output, y_batch)
                        for name, value in batch_metrics.items():
                            train_metrics[name] += value

                train_loss /= len(train_loader)
                for name in train_metrics.keys():
                    train_metrics[name] /= len(train_loader)

                # Calculate per-sample MSE for training set visualization
                train_mse_per_sample_list = []
                train_outputs_list = []
                train_targets_list = []

                # Use multiple batches for more representative visualization
                with torch.no_grad():
                    train_iter = iter(train_loader)
                    for _ in range(min(3, len(train_loader))):  # Use up to 3 batches
                        try:
                            train_x_batch, train_y_batch = next(train_iter)
                            train_x_batch = train_x_batch.to(self.device)
                            train_y_batch = train_y_batch.to(self.device)
                            train_output = self.model(train_x_batch)

                            # Calculate MSE per sample
                            batch_mse_per_sample = torch.mean((train_output - train_y_batch) ** 2, dim=1)

                            train_mse_per_sample_list.append(batch_mse_per_sample)
                            train_outputs_list.append(train_output)
                            train_targets_list.append(train_y_batch)
                        except StopIteration:
                            break

                # Concatenate results from all batches
                if train_mse_per_sample_list:
                    train_mse_per_sample = torch.cat(train_mse_per_sample_list)
                    train_output = torch.cat(train_outputs_list)
                    train_y_batch = torch.cat(train_targets_list)

                    # Get best and worst indices
                    train_best_mse_idx = torch.argmin(train_mse_per_sample)
                    train_worst_mse_idx = torch.argmax(train_mse_per_sample)

                val_loss, val_metrics = self.evaluate(val_loader)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["learning_rate"].append(current_lr)

                for name, value in train_metrics.items():
                    history[f"train_{name}"].append(value)
                for name, value in val_metrics.items():
                    history[f"val_{name}"].append(value)

                epoch_time = time.time() - epoch_start_time
                log_message = (f"Epoch {epoch + 1}/{n_epochs} "
                               f"[{epoch_time:.2f}s]: "
                               f"Train Loss: {train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, "
                               f"LR: {current_lr:.6f}")

                # Add metrics to log message
                for name, value in train_metrics.items():
                    log_message += f", Train {name}: {value:.4f}"
                for name, value in val_metrics.items():
                    log_message += f", Val {name}: {value:.4f}"

                logger.info(log_message)

                # Calculate per-sample MSE for validation set visualization
                val_mse_per_sample_list = []
                val_outputs_list = []
                val_targets_list = []

                # Use a few batches for more representative visualization
                with torch.no_grad():
                    val_iter = iter(val_loader)
                    for _ in range(min(3, len(val_loader))):  # Use up to 3 batches
                        try:
                            x_batch, y_batch = next(val_iter)
                            x_batch = x_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            output = self.model(x_batch)

                            # Calculate MSE per sample
                            mse_per_sample = torch.mean((output - y_batch) ** 2, dim=1)

                            val_mse_per_sample_list.append(mse_per_sample)
                            val_outputs_list.append(output)
                            val_targets_list.append(y_batch)
                        except StopIteration:
                            break

                # Concatenate results from all batches
                if val_mse_per_sample_list:
                    mse_per_sample = torch.cat(val_mse_per_sample_list)
                    output = torch.cat(val_outputs_list)
                    y_batch = torch.cat(val_targets_list)

                    # Get best and worst indices
                    best_mse_idx = torch.argmin(mse_per_sample)
                    worst_mse_idx = torch.argmax(mse_per_sample)

                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    # Training set best and worst RMSE
                    "train_best_pred": train_output[train_best_mse_idx].cpu().detach(),
                    "train_best_true": train_y_batch[train_best_mse_idx].cpu().detach(),
                    "train_best_rmse": torch.sqrt(train_mse_per_sample[train_best_mse_idx]).item(),
                    "train_worst_pred": train_output[train_worst_mse_idx].cpu().detach(),
                    "train_worst_true": train_y_batch[train_worst_mse_idx].cpu().detach(),
                    "train_worst_rmse": torch.sqrt(train_mse_per_sample[train_worst_mse_idx]).item(),
                    # Validation set best and worst RMSE
                    "val_best_pred": output[best_mse_idx].cpu().detach(),
                    "val_best_true": y_batch[best_mse_idx].cpu().detach(),
                    "val_best_rmse": torch.sqrt(mse_per_sample[best_mse_idx]).item(),
                    "val_worst_pred": output[worst_mse_idx].cpu().detach(),
                    "val_worst_true": y_batch[worst_mse_idx].cpu().detach(),
                    "val_worst_rmse": torch.sqrt(mse_per_sample[worst_mse_idx]).item()
                }

                for name, value in train_metrics.items():
                    log_dict[f"train_{name}"] = value
                for name, value in val_metrics.items():
                    log_dict[f"val_{name}"] = value

                # Only log to wandb if it was successfully initialized
                if self.wandb_initialized and epoch %20 == 0:
                    # Create plots for best and worst predictions
                    train_best_fig = create_prediction_plot(
                        train_y_batch[train_best_mse_idx], 
                        train_output[train_best_mse_idx],
                        f"Training Best Prediction (RMSE: {torch.sqrt(train_mse_per_sample[train_best_mse_idx]).item():.4f})"
                    )

                    train_worst_fig = create_prediction_plot(
                        train_y_batch[train_worst_mse_idx], 
                        train_output[train_worst_mse_idx],
                        f"Training Worst Prediction (RMSE: {torch.sqrt(train_mse_per_sample[train_worst_mse_idx]).item():.4f})"
                    )

                    val_best_fig = create_prediction_plot(
                        y_batch[best_mse_idx], 
                        output[best_mse_idx],
                        f"Validation Best Prediction (RMSE: {torch.sqrt(mse_per_sample[best_mse_idx]).item():.4f})"
                    )

                    val_worst_fig = create_prediction_plot(
                        y_batch[worst_mse_idx], 
                        output[worst_mse_idx],
                        f"Validation Worst Prediction (RMSE: {torch.sqrt(mse_per_sample[worst_mse_idx]).item():.4f})"
                    )

                    # Log metrics and plots
                    wandb.log({
                        **log_dict,
                        "train_best_plot": wandb.Image(train_best_fig),
                        "train_worst_plot": wandb.Image(train_worst_fig),
                        "val_best_plot": wandb.Image(val_best_fig),
                        "val_worst_plot": wandb.Image(val_worst_fig)
                    })

                    # Close figures to free memory
                    plt.close(train_best_fig)
                    plt.close(train_worst_fig)
                    plt.close(val_best_fig)
                    plt.close(val_worst_fig)

                self._save_checkpoint(epoch + 1, train_loss, val_loss)

                # Early stopping
                if early_stop is not None and early_stop.step(val_loss, self.model):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            # Training complete
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f}s")

            # Evaluate on test set if available
            if test_loader is not None:
                logger.info("Evaluating on test set")
                test_loss, test_metrics = self.evaluate(test_loader, prefix="Test")

                # Log test metrics
                log_message = f"Test Loss: {test_loss:.4f}"
                for name, value in test_metrics.items():
                    log_message += f", Test {name}: {value:.4f}"

                logger.info(log_message)

                # Calculate per-sample MSE for test set visualization
                test_mse_per_sample_list = []
                test_outputs_list = []
                test_targets_list = []

                # Use multiple batches for more representative visualization
                with torch.no_grad():
                    test_iter = iter(test_loader)
                    for _ in range(min(5, len(test_loader))):  # Use up to 5 batches for test set
                        try:
                            x_batch, y_batch = next(test_iter)
                            x_batch = x_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            output = self.model(x_batch)

                            # Calculate MSE per sample
                            batch_mse_per_sample = torch.mean((output - y_batch) ** 2, dim=1)

                            test_mse_per_sample_list.append(batch_mse_per_sample)
                            test_outputs_list.append(output)
                            test_targets_list.append(y_batch)
                        except StopIteration:
                            break

                # Concatenate results from all batches
                if test_mse_per_sample_list:
                    test_mse_per_sample = torch.cat(test_mse_per_sample_list)
                    output = torch.cat(test_outputs_list)
                    y_batch = torch.cat(test_targets_list)

                    # Get best and worst indices
                    test_best_mse_idx = torch.argmin(test_mse_per_sample)
                    test_worst_mse_idx = torch.argmax(test_mse_per_sample)

                log_dict = {
                    "test_loss": test_loss,
                    # Best case test predictions and targets
                    "test_best_pred": output[test_best_mse_idx].cpu().detach(),
                    "test_best_true": y_batch[test_best_mse_idx].cpu().detach(),
                    "test_best_rmse": torch.sqrt(test_mse_per_sample[test_best_mse_idx]).item(),
                    # Worst case test predictions and targets
                    "test_worst_pred": output[test_worst_mse_idx].cpu().detach(),
                    "test_worst_true": y_batch[test_worst_mse_idx].cpu().detach(),
                    "test_worst_rmse": torch.sqrt(test_mse_per_sample[test_worst_mse_idx]).item()
                }

                for name, value in test_metrics.items():
                    log_dict[f"test_{name}"] = value

                # Only log to wandb if it was successfully initialized
                if self.wandb_initialized:
                    # Create plots for best and worst test predictions
                    test_best_fig = create_prediction_plot(
                        y_batch[test_best_mse_idx], 
                        output[test_best_mse_idx],
                        f"Test Best Prediction (RMSE: {torch.sqrt(test_mse_per_sample[test_best_mse_idx]).item():.4f})"
                    )

                    test_worst_fig = create_prediction_plot(
                        y_batch[test_worst_mse_idx], 
                        output[test_worst_mse_idx],
                        f"Test Worst Prediction (RMSE: {torch.sqrt(test_mse_per_sample[test_worst_mse_idx]).item():.4f})"
                    )

                    # Log metrics and plots
                    wandb.log({
                        **log_dict,
                        "test_best_plot": wandb.Image(test_best_fig),
                        "test_worst_plot": wandb.Image(test_worst_fig)
                    })

                    # Close figures to free memory
                    plt.close(test_best_fig)
                    plt.close(test_worst_fig)

            return history

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise RuntimeError(f"Training failed: {e}")

    def evaluate(self, data_loader: DataLoader, prefix: str="Val"):
        self.model.eval()
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metrics.keys()}

        eval_iter = tqdm(data_loader, desc=f"{prefix} Evaluation")

        with torch.no_grad():
            for x_batch, y_batch in eval_iter:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = self.loss_fn(outputs, y_batch)

                total_loss += loss.item()

                if self.metrics:
                    batch_metrics = self._compute_metrics(outputs, y_batch)
                    for name, value in batch_metrics.items():
                        total_metrics[name] += value

        avg_loss = total_loss / len(data_loader)
        avg_metrics = {name: value / len(data_loader) for name, value in total_metrics.items()}

        return avg_loss, avg_metrics

    def predict(self, data_loader: DataLoader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                preds.append(outputs.cpu().numpy())
        return torch.cat(preds)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def main():
    import argparse
    from src.utils.config_util import Config

    parser = argparse.ArgumentParser(description="Train a TARDIS emulator model")
    parser.add_argument("--config", type=str, default="config/training_configs/training_config.yaml",
                        help="Path to the training configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")

    # Add grid search parameters
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Number of hidden layers (depth)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Width of each hidden layer")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--activation", type=str, default=None,
                        help="Activation function (relu, tanh, sigmoid, mish, etc.)")

    args = parser.parse_args()

    if all([args.n_layers, args.hidden_dim, args.lr, args.activation]):
        config = Config("dummy_path", auto_load=False)
        model_config_path, training_config_path = config.make_config(
            depth=args.n_layers,
            width=args.hidden_dim,
            learning_rate=args.lr,
            activation=args.activation
        )

        config_path = training_config_path
        print(f"Created config files:\n  Model: {model_config_path}\n  Training: {training_config_path}")
    else:
        config_path = args.config
        print(f"Using existing config: {config_path}")

    trainer = Trainer(config_path)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    history = trainer.train()

    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
