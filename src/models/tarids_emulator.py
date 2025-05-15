from typing import Optional

import torch
import yaml
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
from src.data_processing.vae_data_transform import inverse_transform
import numpy as np


class Emulator(nn.Module):
    def __init__(self, config_arch: str = "config.yaml", config_training: str = "traininig.yaml"):
        super().__init__()

        with open(config_arch, "r") as f:
            self.config: dict = yaml.load(f, Loader=yaml.FullLoader)

        model_arch: dict = self.config["architecture"]

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

if __name__ == '__main__':
    model = Emulator("../../config/model_configs/ffn_model.yaml")