from packages import *

def random_subsample_fn(X: np.ndarray, size: int) -> np.ndarray:
    """Selects lines in the input sample randomly.

    Args:
        X(np.ndarray): input sample
        size(int): number of lines to select

    Returns:
        np.ndarray: array of selected lines
    """
    indices = np.random.choice(len(X), size=size, replace=False)
    return indices

def mmd_subsample_fn(X: np.ndarray, size: int) -> np.ndarray:
    """Selects lines in the input sample by greedily minimizing the maximum mena discrepancy.

    Args:
        X(np.ndarray): input sample
        size(int): number of lines to select

    Returns:
        np.ndarray: array of selected lines
    """

    n = X.shape[0]
    assert size<=n

    idx = np.zeros(size, dtype=np.int64)

    k0 = np.zeros((n,size))
    k0[:,0] = 2.0*np.sqrt(np.sum(X.T*X.T, axis=0))

    dist_matrix = cdist(X,X)
    gram_matrix = np.linalg.norm(X, axis=1)[:,None] + np.linalg.norm(X, axis=1)[None] - dist_matrix
    k0_mean = np.mean(gram_matrix, axis=1)

    idx[0] = np.argmin(k0[:,0]-2.0*k0_mean)
    for i in range(1,size):
        x_ = X[idx[i-1]][None]
        dist = (X-x_).T
        k0[:,i] = -np.sqrt(np.sum(dist*dist, axis=0)) + np.sqrt(np.sum(x_.T*x_.T, axis=0)) + np.sqrt(np.sum(X.T*X.T, axis=0))
        idx[i] = np.argmin(k0[:,0] + 2.0*np.sum(k0[:,1:(i+1)], axis=1) - 2.0*(i+1)*k0_mean)
    return idx

def select_subsampling_points(blades: np.ndarray, subsample_size: int, subsample_method) -> np.ndarray:
    """From a list of blades, select the points to keep for each blade.

    Args:
        problem(list): list of blade to consider
        path(str): add to the path

    Returns:
        np.ndarray: Array of np.arrays. Each np.array represents the indices of points to keep for the blade.
    """
    list_of_indices = []
    if subsample_method is not None:
        for blade in blades:
            indices = subsample_method(blade, size = subsample_size)
            list_of_indices.append(indices)
    else:
        pass
    return np.array(list_of_indices)

def subsampling_blades(blades: np.ndarray, subsampling_points: np.ndarray, subsampling_size: int) -> np.ndarray:
    """For a list of blades, subsamples each blade according to the points to keep.

    Args:
        blades(np.ndarray): Array of blades to consider.
        subsampling_points(np.ndarray): Array of points to consider.
        subsampling_size(int): Number of points to keep from each subsampled blade.

    Returns:
        np.ndarray: Array of np.arrays. Each np.array represents a blade subsampled.
    """
    if subsampling_size > subsampling_points.shape[1]:
        raise ValueError("subsampling_size cannot exceed the length of subsampling_points")

    subsampled_blades = []
    for points in subsampling_points:
        subsampled_blades.append(points[:subsampling_size])
    return np.array([blade[subsampled_indices] for blade, subsampled_indices in zip(blades, subsampled_blades)])
