import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    cov= np.cov(data.T)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    comp= eigvec[:,order[:k]]
    for i in range(k):
        v = comp[:, i]
        if v[np.abs(v) > 1e-12][0] < 0:
            comp[:, i] = -v
    return np.round(comp, 4)
    
