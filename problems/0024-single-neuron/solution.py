import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE

	#we are going to have data of shape (N,D), output of size 2, we need out = y = sigmoid (z), z = W @ x +b  where w= 2 x D and b is 2, 

    w= torch.tensor(weights, dtype=torch.float)
    
    X= torch.tensor(features ,dtype=torch.float)
    y= torch.tensor(labels,  dtype=torch.float)
    z= X @ w + bias
    sig = torch.sigmoid(z)
    mse = F.mse_loss(sig, y)
    return sig.tolist(), mse.item()
	
