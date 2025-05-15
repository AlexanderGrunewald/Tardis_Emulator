"""
Tests for the VAE data pipeline module.
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
import importlib.util
from src.data_processing.vae_data_pipeline import create_vae_data_pipeline

# Check if pyarrow is installed
pyarrow_installed = importlib.util.find_spec("pyarrow") is not None

@pytest.fixture
def sample_hdf_file():
    """Create a temporary HDF5 file with test data."""
    tmp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp_file.close()

    try:
        # Create test data
        parameters = pd.DataFrame({
            'param1': [1, 2, 3],
            'param2': [4, 5, 6]
        })
        latents = pd.DataFrame({
            'latent1': [0.1, 0.2, 0.3],
            'latent2': [0.4, 0.5, 0.6]
        })

        # Save to HDF5
        with pd.HDFStore(tmp_file.name, mode='w') as store:
            store.put('parameters', parameters)
            store.put('latents', latents)
            store.put('/model_0/w', pd.Series([1.0, 2.0, 3.0]))
            store.put('/model_0/v_inner', pd.Series([0.1, 0.2, 0.3]))
            store.put('/model_0/v_outer', pd.Series([1.1, 1.2, 1.3]))

        yield tmp_file.name
    finally:
        # Clean up
        if os.path.exists(tmp_file.name):
            try:
                os.remove(tmp_file.name)
            except PermissionError:
                # If we can't remove it now, it will be removed on next test run
                pass


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
def test_create_vae_data_pipeline(sample_hdf_file):
    """Test the create_vae_data_pipeline function."""
    # Create a temporary output file
    tmp_out = tempfile.NamedTemporaryFile(suffix='.feather', delete=False)
    tmp_out.close()
    output_path = tmp_out.name

    scaler_path = os.path.join(tempfile.gettempdir(), "test_scalers.pkl")

    try:
        # Run the pipeline
        df, metadata = create_vae_data_pipeline(
            sample_hdf_file,
            output_path,
            n_elem=5,
            workers=2,
            scale_data=True,
            save_scaler=True,
            scaler_path=scaler_path
        )

        # Check that the output file exists
        assert os.path.exists(output_path)

        # Check that the DataFrame has the expected structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check that the metadata has the expected keys
        assert "input_file" in metadata
        assert "output_file" in metadata
        assert "n_elem" in metadata
        assert "workers" in metadata
        assert "data_type" in metadata
        assert "w_shape" in metadata
        assert "v_shape" in metadata
        assert "scaled" in metadata

        # Check that we can read the feather file
        df_read = pd.read_feather(output_path)
        assert df_read.equals(df)

        # Check that the scaler file was created
        assert os.path.exists(scaler_path)

    finally:
        # Clean up
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
        except PermissionError:
            # If we can't remove it now, it will be removed on next test run
            pass


@pytest.mark.skipif(not pyarrow_installed, reason="pyarrow not installed")
def test_create_vae_data_pipeline_no_scale(sample_hdf_file):
    """Test the create_vae_data_pipeline function without scaling."""
    # Create a temporary output file
    tmp_out = tempfile.NamedTemporaryFile(suffix='.feather', delete=False)
    tmp_out.close()
    output_path = tmp_out.name

    scaler_path = os.path.join(tempfile.gettempdir(), "test_scalers_no_scale.pkl")

    try:
        # Run the pipeline without scaling
        df, metadata = create_vae_data_pipeline(
            sample_hdf_file,
            output_path,
            n_elem=5,
            workers=2,
            scale_data=False,
            save_scaler=False,
            scaler_path=scaler_path
        )

        # Check that the output file exists
        assert os.path.exists(output_path)

        # Check that the DataFrame has the expected structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check that the metadata indicates no scaling
        assert metadata["data_type"] == "raw"
        assert metadata["scaled"] is False
        assert metadata["scaler_path"] is None

        # Check that the scaler file was not created
        assert not os.path.exists(scaler_path)

    finally:
        # Clean up
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
        except PermissionError:
            # If we can't remove it now, it will be removed on next test run
            pass


def test_create_vae_data_pipeline_file_not_found():
    """Test the create_vae_data_pipeline function with a non-existent input file."""
    with pytest.raises(FileNotFoundError):
        create_vae_data_pipeline(
            "non_existent_file.h5",
            "output.feather"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
