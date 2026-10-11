import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    idcs= np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idcs)
    fold_sizes= [n_samples // k +(1 if i<n_samples%k else 0) for i in range(k)]
    splits= [] 
    start = 0 
    for size in fold_sizes:
        stop = start +size 
        test = idcs[start:stop]
        train = np.concatenate([idcs[:start], idcs[stop:]])
        splits.append((train.tolist(),test.tolist()))
        start =stop 
    return splits
