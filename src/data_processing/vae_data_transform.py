from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_vae_config(path_dir: str)->pd.DataFrame:
    with pd.HDFStore(path_dir, mode='r') as configs:
        result = pd.concat([configs["parameters"], configs["latents"]], axis=1)
    return result

def load_vae_latents(path_dir: str)->Tuple[List, List, List]:
    """
    Load VAE latent vectors from an HDF file.

    Args:
        path_dir: Path to the HDF file containing VAE latent vectors

    Returns:
        Tuple of (w_vecs, v_inner_vecs, v_outer_vecs) where each is a list of arrays
    """
    with pd.HDFStore(path_dir, mode='r') as vae_latents:
        vae_latents_keys = vae_latents.keys()
        vae_latents_keys.reverse()

        w_keys = [key for key in vae_latents_keys if str(key).endswith("w")]
        v_inner = [key for key in vae_latents_keys if "v_inner" in key in key]
        v_outer = [key for key in vae_latents_keys if "v_outer" in key in key]

        # Extract data directly as arrays instead of nested lists
        w_vecs = [vae_latents[key].values for key in w_keys]
        v_inner_vecs = [vae_latents[key].values for key in v_inner]
        v_outer_vecs = [vae_latents[key].values for key in v_outer]

    return w_vecs, v_inner_vecs, v_outer_vecs

def interpolate_to_length(data, length)->np.ndarray:
    if data.shape[0] == 0:
        raise ValueError("No data to interpolate")
    if length <= 0:
        raise ValueError("Length of data to interpolate is 0 or less")

    old_indices = np.arange(len(data))
    new_indices = np.linspace(0, len(data)-1, length)
    return np.interp(new_indices, old_indices, data)

def interpolate_data(v_inner, v_outer, w_data, n_elem: int) -> Tuple[np.array, np.array]:
    """
    Interpolate VAE latent vectors to a specified length.

    Args:
        v_inner: Inner velocity vector
        v_outer: Outer velocity vector
        w_data: W vector data
        n_elem: Number of elements to interpolate to

    Returns:
        Tuple of (w_interp, v) where:
            w_interp: Interpolated w vector
            v: Processed velocity vector

    Raises:
        ValueError: If input data is empty or invalid
    """
    try:
        # Convert inputs to numpy arrays and flatten
        v_inner = np.array(v_inner).reshape(-1, 1).flatten()
        v_outer = np.array(v_outer).reshape(-1, 1).flatten()
        w_data = np.array(w_data).reshape(-1, 1).flatten()

        # Validate inputs
        if v_inner.size == 0 or v_outer.size == 0 or w_data.size == 0:
            raise ValueError("Input arrays cannot be empty")

        # Interpolate data to specified length
        v_inner_interp = interpolate_to_length(v_inner, n_elem)
        v_outer_interp = interpolate_to_length(v_outer, n_elem)
        w_interp = interpolate_to_length(w_data, n_elem)

        # Calculate velocity difference
        dv = (v_outer_interp - v_inner_interp)[:-1]

        # Concatenate to form final v vector
        v = np.concatenate((v_inner[:1], dv))  # revert back to v_inner using np.cumsum on v

        return w_interp, v
    except Exception as e:
        print(f"Error in interpolate_data: {e}")
        raise

def execution(v_inner_list, v_outer_list, w_data_list, n_elem: int) -> List[Tuple[np.array, np.array]]:
    """
    Process data items in to interpolate.

    Args:
        v_inner_list: List of v_inner data arrays
        v_outer_list: List of v_outer data arrays
        w_data_list: List of w_data arrays
        n_elem: Number of elements to interpolate to

    Returns:
        List of tuples containing (w_interp, v) for each input data item
    """
    # Create a list of n_elem values with the same length as the input lists

    results = []

    try:
        for i  in range(len(v_inner_list)):
            interp = interpolate_data(v_inner_list[i], v_outer_list[i],
                                      w_data_list[i], n_elem)
            results.append(interp)

    except Exception as e:
        print(f"Error during parallel execution: {e}")
        raise

    return results

def process_vae_data(data_path: str, n_elem: int) -> Tuple[np.array, np.array]:
    """
    High-level function to load and process VAE data.

    Args:
        data_path: Path to the HDF file containing VAE latent vectors
        n_elem: Number of elements to interpolate to

    Returns:
        List of tuples containing (w_interp, v) for each input data item

    Example:
        >>> results = process_vae_data("path/to/vae_data.hdf",n_elem=100)
        >>> w_interp, v = results[0]  # Get the first processed item
    """
    try:
        # Load the VAE latent vectors
        w_vecs, v_inner_vecs, v_outer_vecs = load_vae_latents(data_path)

        # Check that all lists have the same length
        if not (len(w_vecs) == len(v_inner_vecs) == len(v_outer_vecs)):
            raise ValueError("Input data lists must have the same length")

        # Process the data in parallel
        results = execution(v_inner_vecs, v_outer_vecs, w_vecs, n_elem)

        w_interp, v_interp = zip(*results)

        return w_interp, v_interp
    except Exception as e:
        print(f"Error processing VAE data: {e}")
        raise

def scale_results(v: np.array, w: np.array, save_scaler = True, scaler_path: str = "scalers.pkl")->Tuple[np.array, np.array]:
    """
    Scale the results using StandardScaler.

    Args:
        v: Array of v values to scale
        w: Array of w values to scale
        save_scaler: Whether to save the scaler (default: True)
        scaler_path: Path to save the scaler (default: "scalers.pkl")

    Returns:
        Tuple of (v_scaled, w_scaled) arrays
    """
    log_v = np.log(v)
    log_w = np.log(w)

    scaler_v = StandardScaler()
    scaler_w = StandardScaler()

    scaler_v.fit(log_v)
    scaler_w.fit(log_w)

    if save_scaler:
        scalers = {
            'scaler_v': scaler_v,
            'scaler_w': scaler_w
        }
        with open(scaler_path, "wb") as f:  # Note: use "wb" for binary write mode
            pkl.dump(scalers, f)

    return scaler_v.transform(log_v), scaler_w.transform(log_w)


def inverse_transform(v_scaled: np.array = None, w_scaled: np.array = None,
                      scaler_path: str = "scalers.pkl") -> Tuple[np.array, np.array]:
    """
    Inverse transform the scaled data back to original space.

    Args:
        v_scaled: Scaled v values to inverse transform
        w_scaled: Scaled w values to inverse transform
        scaler_path: Path to the saved scaler file

    Returns:
        Tuple of (v_original, w_original) arrays. If an input is None,
        its corresponding output will also be None.
    """
    # Load scalers
    try:
        with open(scaler_path, "rb") as f:
            scalers = pkl.load(f)
            scaler_v = scalers['scaler_v']
            scaler_w = scalers['scaler_w']
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    except KeyError:
        raise KeyError("Invalid scaler file format")

    # Initialize return values
    v_original = None
    w_original = None

    # Inverse transform v if provided
    if v_scaled is not None:
        v_log = scaler_v.inverse_transform(v_scaled)
        v_original = np.cumsum(np.exp(v_log), axis =1)

    # Inverse transform w if provided
    if w_scaled is not None:
        w_log = scaler_w.inverse_transform(w_scaled)
        w_original = np.exp(w_log)

    return v_original, w_original
