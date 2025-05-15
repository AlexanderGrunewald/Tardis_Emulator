import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from src.data_processing.vae_data_transform import (
    load_vae_config,
    load_vae_latents,
    interpolate_to_length,
    interpolate_data,
    thread_execution,
    process_vae_data_parallel
)

@pytest.fixture
def sample_hdf_file():
    """Create a temporary HDF5 file with test data."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
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
        store = pd.HDFStore(tmp.name)
        store.put('parameters', parameters)
        store.put('latents', latents)
        store.put('/model_0/w', pd.Series([1.0, 2.0, 3.0]))
        store.put('/model_0/v_inner', pd.Series([0.1, 0.2, 0.3]))
        store.put('/model_0/v_outer', pd.Series([1.1, 1.2, 1.3]))
        store.close()

        yield tmp.name
    tmp.close()


def test_load_vae_config(sample_hdf_file):
    """Test loading VAE configuration from HDF5 file."""
    config_df = load_vae_config(sample_hdf_file)

    assert isinstance(config_df, pd.DataFrame)
    assert config_df.shape == (3, 4)  # 3 rows, 4 columns (2 params + 2 latents)
    assert all(col in config_df.columns for col in ['param1', 'param2', 'latent1', 'latent2'])
    assert config_df['param1'].iloc[0] == 1
    assert config_df['latent2'].iloc[2] == 0.6


def test_load_vae_latents(sample_hdf_file):
    """Test loading VAE latents from HDF5 file."""
    w_vecs, v_inner_vecs, v_outer_vecs = load_vae_latents(sample_hdf_file)

    # Check that we got lists of arrays
    assert isinstance(w_vecs, list)
    assert isinstance(v_inner_vecs, list)
    assert isinstance(v_outer_vecs, list)

    # Check the content of the vectors
    assert len(w_vecs) == 1
    assert len(v_inner_vecs) == 1
    assert len(v_outer_vecs) == 1

    # Check that each item is a numpy array
    assert isinstance(w_vecs[0], np.ndarray)
    assert isinstance(v_inner_vecs[0], np.ndarray)
    assert isinstance(v_outer_vecs[0], np.ndarray)

    # Check the values
    assert np.array_equal(w_vecs[0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(v_inner_vecs[0], np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(v_outer_vecs[0], np.array([1.1, 1.2, 1.3]))


def test_interpolate_to_length():
    """Test interpolation function with various inputs."""
    # Test case 1: Basic interpolation
    data = np.array([1, 2, 3, 4])
    result = interpolate_to_length(data, 6)
    assert len(result) == 6
    assert result[0] == 1  # First value should be preserved
    assert result[-1] == 4  # Last value should be preserved

    # Test case 2: Interpolation to smaller length
    data = np.array([1, 2, 3, 4, 5])
    result = interpolate_to_length(data, 3)
    assert len(result) == 3
    assert result[0] == 1
    assert result[-1] == 5

    # Test case 3: Same length (should return same values)
    data = np.array([1, 2, 3])
    result = interpolate_to_length(data, 3)
    assert len(result) == 3
    assert np.allclose(result, data)

    # Test case 4: Handle float values
    data = np.array([1.5, 2.5, 3.5])
    result = interpolate_to_length(data, 5)
    assert len(result) == 5
    assert result[0] == 1.5
    assert result[-1] == 3.5


def test_interpolate_to_length_edge_cases():
    """Test interpolation function with edge cases."""
    # Test case 1: Single value
    data = np.array([1])
    result = interpolate_to_length(data, 3)
    assert len(result) == 3
    assert np.allclose(result, [1, 1, 1])

    # Test case 2: Empty array
    with pytest.raises(ValueError):
        interpolate_to_length(np.array([]), 5)

    # Test case 3: Zero target length
    with pytest.raises(ValueError):
        interpolate_to_length(np.array([1, 2, 3]), 0)

    # Test case 4: Negative target length
    with pytest.raises(ValueError):
        interpolate_to_length(np.array([1, 2, 3]), -1)

def test_interpolate_data():
    """Test the interpolate_data function."""
    # Test case 1: Basic interpolation
    v_inner = np.array([0.1, 0.2, 0.3])
    v_outer = np.array([1.1, 1.2, 1.3])
    w_data = np.array([1.0, 2.0, 3.0])
    n_elem = 5

    w_interp, v = interpolate_data(v_inner, v_outer, w_data, n_elem)

    # Check output shapes
    assert len(w_interp) == n_elem
    assert len(v) == n_elem

    # Check that first values are preserved
    assert w_interp[0] == 1.0
    assert v[0] == 0.1

    # Test case 2: Error handling for empty arrays
    with pytest.raises(ValueError):
        interpolate_data(np.array([]), v_outer, w_data, n_elem)

    with pytest.raises(ValueError):
        interpolate_data(v_inner, np.array([]), w_data, n_elem)

    with pytest.raises(ValueError):
        interpolate_data(v_inner, v_outer, np.array([]), n_elem)


def test_thread_execution():
    """Test the thread_execution function for parallel processing."""
    # Create test data
    v_inner_list = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    v_outer_list = [np.array([1.1, 1.2, 1.3]), np.array([1.4, 1.5, 1.6])]
    w_data_list = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    n_elem = 5
    workers = 2

    # Run the function
    results = thread_execution(v_inner_list, v_outer_list, w_data_list, n_elem, workers)

    # Check that we got the expected number of results
    assert len(results) == 2

    # Check that each result is a tuple of two arrays
    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert len(result[0]) == n_elem
        assert len(result[1]) == n_elem


def test_process_vae_data_parallel(sample_hdf_file):
    """Test the process_vae_data_parallel function."""
    # Run the function with the sample HDF file
    n_elem = 5
    workers = 2

    # This should run without errors
    results = process_vae_data_parallel(sample_hdf_file, n_elem, workers)

    # Check that we got results
    assert isinstance(results, list)
    assert len(results) > 0

    # Check that each result is a tuple of two arrays
    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert len(result[0]) == n_elem
        assert len(result[1]) == n_elem


if __name__ == '__main__':
    pass
