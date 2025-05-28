import pprint
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import torch
from torch import nn

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_util import ModelConfig


class Emulator(nn.Module):
    def __init__(self, config_arch: str = "config.yaml"):
        super().__init__()

        # Load the model configuration
        self.model_config = ModelConfig(config_arch)
        self.config: Dict[str, Any] = self.model_config.get_config()


        # Get the architecture parameters
        model_arch: dict = self.model_config.get_architecture()

        self.input_dim: int = model_arch["input_dim"]
        self.hidden_dim: list = model_arch["hidden_dims"]
        self.output_dim: int = model_arch["output_dim"]

        dimensions = [self.input_dim] + self.hidden_dim
        self.layers = nn.ModuleList()
        self.activation = self._get_activation(model_arch["activation"])

        for i in range(len(dimensions) - 1):
            layer = nn.Linear(dimensions[i], dimensions[i + 1])
            self.layers.append(layer)

        self.output_layer = nn.Linear(dimensions[-1], self.output_dim)

    def _get_activation(self, activation_name: str) -> Optional[nn.Module]:
        """Returns the activation function based on the config string."""
        if activation_name == "None":
            return None
        activation_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'mish': nn.Mish(),
            'gelu': nn.GELU()
        }

        return activation_dict.get(activation_name.lower())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    # Use a path relative to the project root
    config_path = os.path.join("config", "model_configs", "ffn_model.yaml")
    model = Emulator(config_path)
    meta = {
        "Model Input": model.input_dim,
        "Model Output": model.output_dim,
        "Model Architecture": model.config["architecture"],
        "Model Activation": model.activation,
        "Layers": model.layers
    }
    pprint.pprint(meta)
