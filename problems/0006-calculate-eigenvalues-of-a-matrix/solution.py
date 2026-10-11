import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix= np.array(matrix)
	tr = sum([matrix[i,i] for i in range(len(matrix))])
	det = np.linalg.det(matrix)
	dis= tr**2 -4*det
	return [(tr + dis**0.5 )/2 , (tr - dis**0.5 )/2 ]