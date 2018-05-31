import numpy as np


def trajectory_matrix(X, K):
    """Build the K-lag trajectory matrix for the signal X

    Parameters
    ----------
    X: array
        Signal to compute the trajectory matrix
    K: int
        Window for the trajectory matrix. The resulting matrix
        will have len(X)-K x K dimension

    """
    TX = [X[i:i+K] for i in range(len(X)-K)]
    return np.array(TX)
