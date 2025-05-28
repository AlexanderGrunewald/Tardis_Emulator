import os
import logging
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

from src.utils.config_util import TrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:


    def __init__(self, config: TrainingConfig):

        self.config = config
        self.data_params = config.get_data_params()
        self.predictors_scaler = None
        self.targets_scaler = None

    def scale_predictor(self, X: pd.DataFrame)->pd.DataFrame:

        standardizer = StandardScaler().fit(X)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(standardizer, f)

        return standardizer.transform(X)

    def load_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Get the paths from the configuration
        predictors_path = self.config.resolve_path(self.data_params["full_predictors"])
        targets_path = self.config.resolve_path(self.data_params["full_targets"])

        # Load the data
        logger.info(f"Loading predictor data from {predictors_path}")
        predictors_df = pd.read_feather(predictors_path).iloc[:,2:].drop([3621, 3945])
        predictors_df_std = self.scale_predictor(predictors_df)

        logger.info(f"Loading target data from {targets_path}")
        targets_df = pd.read_feather(targets_path)

        return predictors_df_std, targets_df


    def create_train_val_test_split(self, predictors: np.ndarray, targets: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

        # Get the split ratios from the configuration
        split_ratios = self.data_params.get("train_val_test_split", [0.7, 0.15, 0.15])

        # Ensure the ratios sum to 1
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            logger.warning(f"Split ratios do not sum to 1: {split_ratios}. Normalizing.")
            split_ratios = [r / sum(split_ratios) for r in split_ratios]

        # Calculate the number of samples for each split
        n_samples = len(predictors)
        n_train = int(split_ratios[0] * n_samples)
        n_val = int(split_ratios[1] * n_samples)
        n_test = n_samples - n_train - n_val

        # Create indices for the splits
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Create the splits
        splits = {
            "train": (predictors[train_indices], targets[train_indices]),
            "val": (predictors[val_indices], targets[val_indices]),
            "test": (predictors[test_indices], targets[test_indices])
        }

        logger.info(f"Created data splits: train={n_train}, val={n_val}, test={n_test}")

        return splits

    def create_dataloaders(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]], batch_size: Optional[int] = None) -> Dict[str, DataLoader]:

        if batch_size is None:
            batch_size = self.config.get_nested_value(["training", "batch_size"], 32)

        dataloaders = {}

        for split_name, (X, y) in splits.items():
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            # Create a TensorDataset
            dataset = TensorDataset(X_tensor, y_tensor)

            # Create a DataLoader
            shuffle = (split_name == "train")  # Only shuffle the training data
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Use 0 workers for simplicity
                pin_memory=torch.cuda.is_available()  # Pin memory if using CUDA
            )

            logger.info(f"Created {split_name} DataLoader with {len(dataset)} samples")

        return dataloaders

    def prepare_data(self, batch_size: Optional[int] = None) -> Dict[str, DataLoader]:

        # Load the data
        predictors_df, targets_df = self.load_preprocessed_data()

        # Check if the number of samples match
        n_predictors = len(predictors_df)
        n_targets = len(targets_df)

        if n_predictors != n_targets:
            logger.warning(f"Mismatch in number of samples: {n_predictors} predictors vs {n_targets} targets")

        # Split the data
        splits = self.create_train_val_test_split(predictors_df, targets_df.values)

        # Create DataLoader objects
        dataloaders = self.create_dataloaders(splits, batch_size)

        return dataloaders
