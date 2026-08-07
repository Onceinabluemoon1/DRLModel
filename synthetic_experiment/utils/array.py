import os

import numpy as np


def _save_array(arr: np.ndarray, path: str) -> None:
    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    if path.endswith(".npy"):
        np.save(path, arr)
    elif path.endswith(".csv"):
        np.savetxt(path, arr, delimiter=",")
    else:
        # Default to .npy if extension is unrecognized
        np.save(path + ".npy", arr)


def _load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Unsupported file extension. Use .npy or .csv.")
