"""
VAE Data Pipeline

This module provides a pipeline for processing VAE data and saving it to a feather file.
The pipeline uses functions from the vae_data_transform module to process the data.
"""

import os
import sys
import logging
import importlib.util
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Check if pyarrow is installed
pyarrow_installed = importlib.util.find_spec("pyarrow") is not None

from src.data_processing.vae_data_transform import (
    process_vae_data,
    scale_results,
    load_vae_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_vae_data_pipeline(
    input_hdf_path: str,
    config_path: str,
    output_targets_path: str,
    output_predictors_path: str,
    n_elem: int = 100,
    scale_data: bool = True,
    save_scaler: bool = True,
    scaler_path: str = "scalers.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Process VAE data and save it to feather files.

    Args:
        input_hdf_path: Path to the input HDF file containing VAE latent vectors
        config_path: Path to the config HDF file containing latent values
        output_targets_path: Path to save the targets (w and v values) feather file
        output_predictors_path: Path to save the predictors (latent values) feather file
        n_elem: Number of elements to interpolate to (default: 100)
        scale_data: Whether to scale the data (default: True)
        save_scaler: Whether to save the scaler (default: True)
        scaler_path: Path to save the scaler (default: "scalers.pkl")

    Returns:
        Tuple of (targets_df, predictors_df, metadata) where:
            targets_df: DataFrame containing the target data (w and v values)
            predictors_df: DataFrame containing the predictor data (latent values)
            metadata: Dictionary containing metadata about the processing

    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If the input data is invalid
        IOError: If there is an error writing the output files
    """
    logger.info(f"Starting VAE data pipeline with input: {input_hdf_path}")

    try:
        # Check if input file exists
        if not os.path.exists(input_hdf_path):
            raise FileNotFoundError(f"Input file not found: {input_hdf_path}")

        # Process the data in parallel
        logger.info(f"Processing VAE data with n_elem={n_elem}")
        w_interp, v_interp = process_vae_data(input_hdf_path, n_elem)
        logger.info("Finished interpolating VAE data")
        logger.info(f"Loading Vae Latent vectors from {config_path}")
        latent_vals = load_vae_config(config_path).reset_index(drop=True)

        # Convert tuples of arrays to numpy arrays
        w_array = np.array(w_interp)
        v_array = np.array(v_interp)

        # Scale the data if requested
        if scale_data:
            logger.info(f"Scaling the data, saving scaler to {scaler_path}")
            w_scaled, v_scaled = scale_results(v_array, w_array, save_scaler=save_scaler, scaler_path=scaler_path)
            w_final = w_scaled
            v_final = v_scaled
            data_type = "scaled"
        else:
            w_final = w_array
            v_final = v_array
            data_type = "raw"

        # Create a DataFrame from the processed data
        logger.info("Creating DataFrame from processed data")

        # Create column names for the DataFrame
        w_cols = [f"w_{i}" for i in range(w_final.shape[1])]
        v_cols = [f"v_{i}" for i in range(v_final.shape[1])]

        n_latents = latent_vals.shape[1] -2 # exclude parameters
        logger.info(f"Creating a DataFrame from {n_latents} latents")

        latent_vals.rename({key:f"latent_{key}" for key in range(n_latents)}, 
                           axis=1, inplace=True)

        # Create DataFrames for w and v data
        w_df = pd.DataFrame(w_final, columns=w_cols)
        v_df = pd.DataFrame(v_final, columns=v_cols)

        # Concatenate the DataFrames for targets (w and v)
        targets_df = pd.concat([w_df, v_df], axis=1)

        # Use latent_vals as predictors DataFrame
        predictors_df = latent_vals

        # Save the targets DataFrame to a feather file
        logger.info(f"Saving targets DataFrame to feather file: {output_targets_path}")
        targets_dir = os.path.dirname(output_targets_path)
        if targets_dir and not os.path.exists(targets_dir):
            os.makedirs(targets_dir)

        # Save the predictors DataFrame to a feather file
        logger.info(f"Saving predictors DataFrame to feather file: {output_predictors_path}")
        predictors_dir = os.path.dirname(output_predictors_path)
        if predictors_dir and not os.path.exists(predictors_dir):
            os.makedirs(predictors_dir)

        if not pyarrow_installed:
            logger.warning("pyarrow not installed, saving as CSV instead of feather")
            # If pyarrow is not installed, save as CSV as a fallback

            # Save targets as CSV
            targets_csv_path = output_targets_path.replace('.feather', '.csv')
            if targets_csv_path == output_targets_path:  # If no .feather extension
                targets_csv_path = output_targets_path + '.csv'
            targets_df.to_csv(targets_csv_path, index=False)
            logger.info(f"Saved targets DataFrame to CSV file: {targets_csv_path}")

            # Save predictors as CSV
            predictors_csv_path = output_predictors_path.replace('.feather', '.csv')
            if predictors_csv_path == output_predictors_path:  # If no .feather extension
                predictors_csv_path = output_predictors_path + '.csv'
            predictors_df.to_csv(predictors_csv_path, index=False)
            logger.info(f"Saved predictors DataFrame to CSV file: {predictors_csv_path}")
        else:
            # Save as feather if pyarrow is installed
            targets_df.to_feather(output_targets_path)
            predictors_df.to_feather(output_predictors_path)

        # Create metadata
        metadata = {
            "input_file": input_hdf_path,
            "config_file": config_path,
            "targets_file": output_targets_path,
            "predictors_file": output_predictors_path,
            "n_elem": n_elem,
            "data_type": data_type,
            "w_shape": w_final.shape,
            "v_shape": v_final.shape,
            "latents_shape": predictors_df.shape,
            "scaled": scale_data,
            "scaler_path": scaler_path if scale_data and save_scaler else None
        }

        logger.info("VAE data pipeline completed successfully")
        return targets_df, predictors_df, metadata

    except Exception as e:
        logger.error(f"Error in VAE data pipeline: {e}")
        raise


def main():
    """
    Main function to run the VAE data pipeline from the command line.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Process VAE data and save to feather files")
    parser.add_argument("input_path", help="Path to the input HDF file")
    parser.add_argument("config_vae", help="Path to the config HDF file")
    parser.add_argument("targets_path", help="Path to save the targets (w and v values) feather file")
    parser.add_argument("predictors_path", help="Path to save the predictors (latent values) feather file")
    parser.add_argument("--n_elem", type=int, default=20, help="Number of elements to interpolate to")
    parser.add_argument("--no-scale", action="store_true", help="Do not scale the data")
    parser.add_argument("--no-save-scaler", action="store_true", help="Do not save the scaler")
    parser.add_argument("--scaler-path", default="scalers.pkl", help="Path to save the scaler")

    args = parser.parse_args()

    targets_df, predictors_df, metadata = create_vae_data_pipeline(args.input_path, args.config_vae, args.targets_path,
                                                                   args.predictors_path, n_elem=args.n_elem,
                                                                   scale_data=not args.no_scale,
                                                                   save_scaler=not args.no_save_scaler,
                                                                   scaler_path=args.scaler_path)

    # Print some information about the processed data
    print(f"Targets shape: {targets_df.shape}")
    print(f"Predictors shape: {predictors_df.shape}")
    print(f"Files saved: {metadata['targets_file']} and {metadata['predictors_file']}")


if __name__ == "__main__":
    main()
