import math
import numpy as np 
def softmax(scores: list[float]) -> list[float]:
    
    scores= np.array(scores)
    scores-= max(scores)
    e=np.exp(scores)
    return e/e.sum()