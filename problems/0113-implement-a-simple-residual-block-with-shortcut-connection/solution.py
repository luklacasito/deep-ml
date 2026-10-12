import torch
import torch.nn.functional as F
def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implement a simple residual block with shortcut connection.
    
    Args:
        x: 1D input tensor
        w1: First weight matrix
        w2: Second weight matrix
    
    Returns:
        Output tensor after residual block processing
    """
    step1 = F.relu(x@ w1)
    step2 = w2@ step1 + x 
    return F.relu(step2)