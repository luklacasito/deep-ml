import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here

    X= torch.tensor(features,  dtype=torch.float)
    y =torch.tensor(labels, dtype=torch.float)
    b= torch.tensor(initial_bias).requires_grad_(True)
    w= torch.tensor(initial_weights, dtype=torch.float).requires_grad_(True)
    mse=[]
    for _ in range(epochs):
        probs= torch.sigmoid (X@ w + b)
        loss = F.mse_loss(probs,y)
        mse.append(round(loss.item(), 4))
        loss.backward()
        with torch.no_grad():
            w-=learning_rate*w.grad
            b-= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return [round(v, 4) for v in w.tolist()], round(b.item(), 4), mse
        

        


    