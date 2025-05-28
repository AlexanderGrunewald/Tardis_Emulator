"""
TARDIS Emulator - Enhanced Trainer module

This module provides an improved version of the Trainer class with better error handling,
flexibility, and additional features for training PyTorch models for the TARDIS emulator.
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Optional imports with error handling
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_util import TrainingConfig
from src.utils.data_manager import DataManager
from src.models.tarids_emulator import Emulator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Attributes:
        patience (int): Number of epochs to wait before stopping if no improvement
        min_delta (float): Minimum change in monitored value to qualify as improvement
        best_loss (float): Best loss value observed so far
        counter (int): Counter for epochs with no improvement
        best_model_path (str): Path to save the best model
        save_best (bool): Whether to save the best model
    """
    def __init__(self, patience: int, min_delta: float = 0.0, 
                 best_model_path: Optional[str] = None):
        """
        Initialize the early stopping handler.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float, optional): Minimum change to qualify as improvement. Defaults to 0.0.
            best_model_path (str, optional): Path to save the best model. Defaults to None.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_path = best_model_path
        self.save_best = best_model_path is not None

    def step(self, loss: float, model: Optional[nn.Module] = None) -> bool:
        """
        Update early stopping state with current loss value.
        
        Args:
            loss (float): Current validation loss
            model (nn.Module, optional): Model to save if this is the best loss. Defaults to None.
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            
            # Save best model if path is provided
            if self.save_best and model is not None:
                self._save_model(model)
                
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False
    
    def _save_model(self, model: nn.Module) -> None:
        """
        Save the model to the specified path.
        
        Args:
            model (nn.Module): Model to save
        """
        if self.best_model_path:
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(model.state_dict(), self.best_model_path)
            logger.info(f"Saved best model to {self.best_model_path}")


class Trainer:
    """
    Enhanced trainer class for PyTorch models.
    
    This class handles the training process including:
    - Loading and managing configuration
    - Setting up model, optimizer, and scheduler
    - Training and validation loops
    - Metrics tracking and logging
    - Model checkpointing
    - Early stopping
    
    Attributes:
        train_config (TrainingConfig): Training configuration
        data_manager (DataManager): Data manager for loading and preprocessing data
        device (torch.device): Device to use for training
        model (nn.Module): PyTorch model to train
        optimizer (optim.Optimizer): Optimizer for training
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler
        loss_fn (Callable): Loss function
        metrics (Dict[str, Callable]): Dictionary of metric functions
        checkpoint_dir (str): Directory to save checkpoints
    """
    def __init__(self, 
                 train_config_path: str, 
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 metrics: Optional[Dict[str, Callable]] = None):
        """
        Initialize the trainer.
        
        Args:
            train_config_path (str): Path to the training configuration file
            model (nn.Module, optional): PyTorch model to train. Defaults to None.
            loss_fn (Callable, optional): Loss function. Defaults to F.mse_loss.
            metrics (Dict[str, Callable], optional): Dictionary of metric functions. Defaults to None.
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        try:
            self.train_config = TrainingConfig(train_config_path)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {train_config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise ValueError(f"Invalid configuration: {e}")
            
        try:
            self.data_manager = DataManager(self.train_config)
        except Exception as e:
            logger.error(f"Error initializing data manager: {e}")
            raise
            
        # Set up device
        self.device = self._setup_device()
        
        # Set up model
        self.model = self._setup_model(model)
        
        # Set up loss function and metrics
        self.loss_fn = loss_fn if loss_fn is not None else F.mse_loss
        self.metrics = metrics if metrics is not None else {}
        
        # Set up optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Set up checkpoint directory
        self.checkpoint_dir = self._setup_checkpoint_dir()
        
        # Initialize wandb if available and enabled
        self._setup_wandb()
        
    def _setup_device(self) -> torch.device:
        """
        Set up the device for training.
        
        Returns:
            torch.device: Device to use for training
        """
        if torch.cuda.is_available():
            logger.info("Using CUDA device.")
            return torch.device('cuda')
        else:
            logger.info("Using CPU device.")
            return torch.device('cpu')
            
    def _setup_model(self, model: Optional[nn.Module]) -> nn.Module:
        """
        Set up the model for training.
        
        Args:
            model (nn.Module, optional): PyTorch model to train. Defaults to None.
            
        Returns:
            nn.Module: Model to train
        """
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

    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create the optimizer based on configuration.
        
        Returns:
            optim.Optimizer: PyTorch optimizer
            
        Raises:
            ValueError: If the optimizer is not supported
        """
        # Get optimizer parameters from config
        train_params = self.train_config.get_training_params()
        lr = train_params.get("learning_rate", 0.001)
        opt_name = train_params.get("optimizer", "adam").lower()
        wd = train_params.get("weight_decay", 0.0)
        
        logger.info(f"Creating optimizer: {opt_name} with lr={lr}, weight_decay={wd}")
        
        # Create optimizer based on name
        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            momentum = train_params.get("momentum", 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        elif opt_name == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create the learning rate scheduler based on configuration.
        
        Returns:
            optim.lr_scheduler._LRScheduler: PyTorch scheduler or None if not configured
        """
        # Get scheduler parameters from config
        train_params = self.train_config.get_training_params()
        scheduler_name = train_params.get("scheduler", "none").lower()
        scheduler_params = train_params.get("scheduler_params", {})
        
        if scheduler_name == "none" or not scheduler_name:
            logger.info("No scheduler configured")
            return None
            
        logger.info(f"Creating scheduler: {scheduler_name}")
        
        # Create scheduler based on name
        if scheduler_name == "reducelronplateau":
            # Note: Changed mode to "min" since we're minimizing loss
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",  # Changed from "max" to "min" for loss minimization
                patience=scheduler_params.get("patience", 10),
                factor=scheduler_params.get("factor", 0.5),
                min_lr=scheduler_params.get("min_lr", 1e-6),
                verbose=True
            )
        elif scheduler_name == "steplr":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get("step_size", 10),
                gamma=scheduler_params.get("gamma", 0.1),
                verbose=True
            )
        elif scheduler_name == "cosineannealinglr":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get("t_max", 10),
                eta_min=scheduler_params.get("eta_min", 0),
                verbose=True
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using no scheduler")
            return None
            
    def _setup_checkpoint_dir(self) -> str:
        """
        Set up the checkpoint directory.
        
        Returns:
            str: Path to the checkpoint directory
        """
        # Get checkpoint directory from config
        logging_params = self.train_config.get_logging_params()
        checkpoint_dir = logging_params.get("checkpoint_dir", "results/checkpoints")
        
        # Resolve path relative to config file
        checkpoint_dir = self.train_config.resolve_path(checkpoint_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir
        
    def _setup_wandb(self) -> None:
        """
        Set up Weights & Biases for experiment tracking if enabled.
        """
        # Get wandb parameters from config
        logging_params = self.train_config.get_logging_params()
        use_wandb = logging_params.get("wandb", False)
        
        if not use_wandb or not WANDB_AVAILABLE:
            if use_wandb and not WANDB_AVAILABLE:
                logger.warning("Weights & Biases is enabled in config but not installed")
            return
            
        # Get project name and run name from config
        project_name = logging_params.get("wandb_project", "tardis-emulator")
        run_name = logging_params.get("wandb_run_name", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Initialize wandb
        try:
            wandb.init(project=project_name, name=run_name, config=self.train_config.get_config())
            logger.info(f"Initialized Weights & Biases: project={project_name}, run={run_name}")
        except Exception as e:
            logger.error(f"Error initializing Weights & Biases: {e}")
            
    def _setup_data(self) -> Dict[str, DataLoader]:
        """
        Set up data loaders for training, validation, and testing.
        
        Returns:
            Dict[str, DataLoader]: Dictionary of data loaders
            
        Raises:
            RuntimeError: If data preparation fails
        """
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
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dict[str, List[float]]: Dictionary of training history
            
        Raises:
            RuntimeError: If training fails
        """
        # Set up data loaders
        try:
            data_loaders = self._setup_data()
            train_loader = data_loaders["train"]
            val_loader = data_loaders["val"]
            test_loader = data_loaders.get("test")  # May not exist
        except Exception as e:
            logger.error(f"Error setting up data loaders: {e}")
            raise RuntimeError(f"Failed to set up data loaders: {e}")
            
        # Get training parameters from config
        train_params = self.train_config.get_training_params()
        n_epochs = train_params.get("num_epochs", 100)
        
        # Set up early stopping
        early_stop_params = train_params.get("early_stopping", {})
        early_stop = None
        
        if early_stop_params.get("enabled", False):
            # Create best model path
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            
            early_stop = EarlyStopping(
                patience=early_stop_params.get("patience", 10),
                min_delta=early_stop_params.get("min_delta", 0.0),
                best_model_path=best_model_path
            )
            logger.info(f"Early stopping enabled with patience={early_stop.patience}")
            
        # Initialize training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        # Add metrics to history
        for metric_name in self.metrics.keys():
            history[f"train_{metric_name}"] = []
            history[f"val_{metric_name}"] = []
            
        # Training loop
        logger.info(f"Starting training for {n_epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(n_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_metrics = {name: 0.0 for name in self.metrics.keys()}
                
                # Create progress bar for training if tqdm is available
                train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]") if TQDM_AVAILABLE else train_loader
                
                for x_batch, y_batch in train_iter:
                    # Move data to device
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(x_batch)
                    loss = self.loss_fn(outputs, y_batch)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    
                    # Compute additional metrics
                    if self.metrics:
                        batch_metrics = self._compute_metrics(outputs, y_batch)
                        for name, value in batch_metrics.items():
                            train_metrics[name] += value
                
                # Calculate average loss and metrics
                train_loss /= len(train_loader)
                for name in train_metrics.keys():
                    train_metrics[name] /= len(train_loader)
                    
                # Validation phase
                val_loss, val_metrics = self.evaluate(val_loader)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                        
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["learning_rate"].append(current_lr)
                
                for name, value in train_metrics.items():
                    history[f"train_{name}"].append(value)
                    
                for name, value in val_metrics.items():
                    history[f"val_{name}"].append(value)
                    
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                log_message = (f"Epoch {epoch+1}/{n_epochs} "
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
                
                # Log to wandb if enabled
                if WANDB_AVAILABLE and wandb.run is not None:
                    log_dict = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr
                    }
                    
                    # Add metrics to wandb log
                    for name, value in train_metrics.items():
                        log_dict[f"train_{name}"] = value
                    for name, value in val_metrics.items():
                        log_dict[f"val_{name}"] = value
                        
                    wandb.log(log_dict)
                    
                # Save checkpoint
                self._save_checkpoint(epoch + 1, train_loss, val_loss)
                
                # Early stopping
                if early_stop is not None and early_stop.step(val_loss, self.model):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
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
                
                # Log to wandb if enabled
                if WANDB_AVAILABLE and wandb.run is not None:
                    log_dict = {"test_loss": test_loss}
                    for name, value in test_metrics.items():
                        log_dict[f"test_{name}"] = value
                    wandb.log(log_dict)
                    
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise RuntimeError(f"Training failed: {e}")
            
    def evaluate(self, data_loader: DataLoader, prefix: str = "Val") -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader (DataLoader): DataLoader for evaluation
            prefix (str, optional): Prefix for logging. Defaults to "Val".
            
        Returns:
            Tuple[float, Dict[str, float]]: Tuple of (loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metrics.keys()}
        
        # Create progress bar for evaluation if tqdm is available
        eval_iter = tqdm(data_loader, desc=f"{prefix} Evaluation") if TQDM_AVAILABLE else data_loader
        
        with torch.no_grad():
            for x_batch, y_batch in eval_iter:
                # Move data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(x_batch)
                loss = self.loss_fn(outputs, y_batch)
                
                # Update loss and metrics
                total_loss += loss.item()
                
                # Compute additional metrics
                if self.metrics:
                    batch_metrics = self._compute_metrics(outputs, y_batch)
                    for name, value in batch_metrics.items():
                        total_metrics[name] += value
                        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(data_loader)
        avg_metrics = {name: value / len(data_loader) for name, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
        
    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            data_loader (DataLoader): DataLoader for prediction
            
        Returns:
            torch.Tensor: Model predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                predictions.append(outputs.cpu())
                
        return torch.cat(predictions, dim=0)
        
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
    """
    Main function to demonstrate the trainer usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a TARDIS emulator model")
    parser.add_argument("--config", type=str, default="config/training_configs/training_config.yaml",
                        help="Path to the training configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    # Train model
    history = trainer.train()
    
    # Print final metrics
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    
if __name__ == "__main__":
    main()