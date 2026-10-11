import math
import numpy as np
def sigmoid(z: float) -> float:
	#Your code here
	

	return (np.exp(z)/(1+np.exp(z))).round(4)