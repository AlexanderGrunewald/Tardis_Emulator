# Emulator Project for TARDIS

A PyTorch-based emulator for the TARDIS radiative transfer code.

## Project Overview

This project aims to develop a neural network emulator for the TARDIS radiative transfer code, allowing for fast approximations of TARDIS simulations by providingit better starting values for type 1a supernovs.

## Project Structure

```
emulatorv3/
│
├── config/                     # Configuration files
│   ├── model_configs/          # Model architecture configurations
│   └── training_configs/       # Training hyperparameters and settings
│
├── data/                       # Data storage
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, transformed data ready for modeling
│   ├── interim/                # Intermediate data that has been transformed
│   └── external/               # Data from external sources
│
├── docs/                       # Documentation
│
├── notebooks/                  # Jupyter notebooks
│   ├── exploratory/            # Exploration and prototyping
│   ├── analysis/               # Analysis of results
│   └── visualization/          # Visualizations for presentations/papers
│
├── results/                    # Experiment results, model checkpoints, figures
│
├── src/                        # Source code
│   ├── models/                 # Model architectures
│   ├── data_processing/        # Data loading and preprocessing
│   ├── training/               # Training scripts and utilities
│   ├── evaluation/             # Model evaluation code
│   └── utils/                  # Utility functions
│
└── tests/                      # Unit tests
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd emulatorv3
   ```

2. Create a virtual environment (optional but recommended):
   ```
   conda create -n tardis_emulator
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Data Preparation

1. Place your raw DALEC VAE simulation data in the `data/raw/` directory slomg with the latent config.
2. Use the data processing scripts in `src/data_processing/` to preprocess the data.

```
python src/data_processing/vae_data_pipeline.py data/raw/VAE_Simulations.hdf data/raw/VAE_Simulation_config.hdf data/processed/target/y_data.feather data/processed/predictor/x_data.feather --n_elem 20
```

#### Model Configuration

1. Configure your model architecture in a YAML file in `config/model_configs/`.
2. Configure training parameters in a YAML file in `config/training_configs/`.

Alternatively you can specify the number of layers, activation functions, and number of neurons manually.

```
python src/training/trainer.py --n_layers 10 --hidden_dim 200 --lr 1e4 --activation mish
```

## License

MIT License

