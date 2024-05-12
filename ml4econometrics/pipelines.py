import numpy as np

def standardize(x):
    """
    standardize:
        For a matrix X of dimension (n, p), makes sure that each column is zero mean and unit variance.

    Args:
        X (np.array): The array to be standardized.
    """
    return (x - x.mean(axis=0)) / x.std(axis=0)