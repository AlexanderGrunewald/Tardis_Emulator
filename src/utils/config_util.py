import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Exception raised for errors in the configuration."""
    pass

class Config:


    def __init__(self, config_file_path: str, auto_load: bool = True):

        self.config_file = config_file_path
        self.config = None
        self.base_dir = os.path.dirname(os.path.abspath(config_file_path))

        if auto_load:
            self.read_config()

    def make_config(self, depth: int, width: int, learning_rate: float, activation: str, 
                  output_dir: str = "config/grid_configs"):

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create model config
        model_config = {
            "model_name": f"FFN_Emulator_d{depth}_w{width}_{activation}",
            "model_type": "FFN",
            "architecture": {
                "input_dim": 6,  # Assuming this is fixed
                "hidden_dims": [width] * depth,
                "output_dim": 40,  # Assuming this is fixed
                "activation": activation,
                "dropout_rate": "None"
            }
        }

        # Create model config filename
        model_config_filename = f"ffn_model_d{depth}_w{width}_{activation}.yaml"
        model_config_path = os.path.join(output_dir, model_config_filename)

        # Save model config
        with open(model_config_path, "w") as f:
            yaml.dump(model_config, f, default_flow_style=False)

        # Create training config
        training_config = {
            "model_config": model_config_path,
            "training": {
                "batch_size": 32,
                "num_epochs": 2000,
                "learning_rate": learning_rate,
                "weight_decay": 0.0001,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "scheduler_params": {
                    "patience": 10,
                    "factor": 0.5,
                    "min_lr": 0.00001
                },
                "early_stopping": {
                    "enabled": True,
                    "patience": 20,
                    "min_delta": 0.0001
                }
            },
            "data": {
                "full_predictors": "../../data/processed/predictor/x_data.feather",
                "full_targets": "../../data/processed/target/y_data.feather",
                "train_val_test_split": [0.7, 0.15, 0.15]
            },
            "logging": {
                "log_dir": "../../results/logs",
                "checkpoint_dir": "../../results/checkpoints",
                "save_best_only": True,
                "save_frequency": 5,
                "wandb": True,
                "wandb_entity": "alexgrunewald123-michigan-state-university",
                "wandb_project": "tardis-emulator",
                "wandb_run_name": f"ffn_d{depth}_w{width}_lr{learning_rate}_{activation}"
            }
        }

        # Create training config filename
        training_config_filename = f"training_config_d{depth}_w{width}_lr{learning_rate}_{activation}.yaml"
        training_config_path = os.path.join(output_dir, training_config_filename)

        # Save training config
        with open(training_config_path, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False)

        return model_config_path, training_config_path



    def read_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_file, "r") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            logger.debug(f"Loaded configuration from {self.config_file}")
            return self.config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {self.config_file}: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        if self.config is None:
            raise ConfigError("Configuration has not been loaded. Call read_config() first.")
        return self.config

    def resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    def get_nested_value(self, keys: Union[str, List[str]], default: Any = None) -> Any:
        if self.config is None:
            raise ConfigError("Configuration has not been loaded. Call read_config() first.")

        if isinstance(keys, str):
            return self.config.get(keys, default)

        current = self.config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


class ModelConfig(Config):

    def __init__(self, config_file_path: str, auto_load: bool = True):
        super().__init__(config_file_path, auto_load)


    def get_architecture(self) -> Dict[str, Any]:
        return self.get_nested_value("architecture", {})


class TrainingConfig(Config):

    def __init__(self, config_file_path: str, auto_load: bool = True):

        super().__init__(config_file_path, auto_load)
        self.model_config = None

        if auto_load and "model_config" in self.config:
            self.load_model_config()


    def load_model_config(self) -> ModelConfig:

        if "model_config" not in self.config:
            raise ConfigError("Model configuration path not specified in training configuration")

        model_config_path = self.resolve_path(self.config["model_config"])
        self.model_config = ModelConfig(model_config_path)
        return self.model_config

    def get_model_config(self) -> ModelConfig:

        if self.model_config is None:
            raise ConfigError("Model configuration has not been loaded. Call load_model_config() first.")
        return self.model_config

    def get_training_params(self) -> Dict[str, Any]:
        """
        Get the training parameters.

        Returns:
            dict: The training parameters
        """
        return self.get_nested_value("training", {})

    def get_data_params(self) -> Dict[str, Any]:
        """
        Get the data parameters.

        Returns:
            dict: The data parameters
        """
        return self.get_nested_value("data", {})

    def get_logging_params(self) -> Dict[str, Any]:
        """
        Get the logging parameters.

        Returns:
            dict: The logging parameters
        """
        return self.get_nested_value("logging", {})
