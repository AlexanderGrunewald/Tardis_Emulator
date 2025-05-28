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
    """
    A general configuration loader for YAML files.

    This class can load configuration files and resolve references to other configuration files.
    It also supports validation of required fields and provides access to the loaded configuration.

    Attributes:
        config_file (str): Path to the configuration file
        config (dict): The loaded configuration
        base_dir (str): The directory containing the configuration file
    """

    def __init__(self, config_file_path: str, auto_load: bool = True):
        """
        Initialize the Config object.

        Args:
            config_file_path (str): Path to the configuration file
            auto_load (bool, optional): Whether to automatically load the configuration. Defaults to True.
            validate (bool, optional): Whether to validate the configuration. Defaults to True.
        """
        self.config_file = config_file_path
        self.config = None
        self.base_dir = os.path.dirname(os.path.abspath(config_file_path))

        if auto_load:
            self.read_config()

    def read_config(self) -> Dict[str, Any]:
        """
        Read the configuration from the file.

        Returns:
            dict: The loaded configuration

        Raises:
            FileNotFoundError: If the configuration file does not exist
            yaml.YAMLError: If the configuration file is not valid YAML
        """
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
        """
        Get the loaded configuration.

        Returns:
            dict: The loaded configuration

        Raises:
            ConfigError: If the configuration has not been loaded
        """
        if self.config is None:
            raise ConfigError("Configuration has not been loaded. Call read_config() first.")
        return self.config

    def resolve_path(self, path: str) -> str:
        """
        Resolve a path relative to the configuration file.

        Args:
            path (str): The path to resolve

        Returns:
            str: The resolved path
        """
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    def get_nested_value(self, keys: Union[str, List[str]], default: Any = None) -> Any:
        """
        Get a nested value from the configuration.

        Args:
            keys (str or list): The key or list of keys to access the nested value
            default (any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            any: The value at the specified keys, or the default value if not found

        Raises:
            ConfigError: If the configuration has not been loaded
        """
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
    """
    Configuration loader for model configuration files.

    This class extends the Config class with specific functionality for model configurations.
    """

    def __init__(self, config_file_path: str, auto_load: bool = True):
        """
        Initialize the ModelConfig object.

        Args:
            config_file_path (str): Path to the model configuration file
            auto_load (bool, optional): Whether to automatically load the configuration. Defaults to True.
        """
        super().__init__(config_file_path, auto_load)


    def get_architecture(self) -> Dict[str, Any]:
        """
        Get the architecture configuration.

        Returns:
            dict: The architecture configuration
        """
        return self.get_nested_value("architecture", {})


class TrainingConfig(Config):
    """
    Configuration loader for training configuration files.

    This class extends the Config class with specific functionality for training configurations,
    including loading the referenced model configuration.
    """

    def __init__(self, config_file_path: str, auto_load: bool = True):
        """
        Initialize the TrainingConfig object.

        Args:
            config_file_path (str): Path to the training configuration file
            auto_load (bool, optional): Whether to automatically load the configuration. Defaults to True.
        """
        super().__init__(config_file_path, auto_load)
        self.model_config = None

        if auto_load and "model_config" in self.config:
            self.load_model_config()


    def load_model_config(self) -> ModelConfig:
        """
        Load the model configuration referenced in the training configuration.

        Returns:
            ModelConfig: The loaded model configuration

        Raises:
            ConfigError: If the model configuration path is not specified
        """
        if "model_config" not in self.config:
            raise ConfigError("Model configuration path not specified in training configuration")

        model_config_path = self.resolve_path(self.config["model_config"])
        self.model_config = ModelConfig(model_config_path)
        return self.model_config

    def get_model_config(self) -> ModelConfig:
        """
        Get the loaded model configuration.

        Returns:
            ModelConfig: The loaded model configuration

        Raises:
            ConfigError: If the model configuration has not been loaded
        """
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
