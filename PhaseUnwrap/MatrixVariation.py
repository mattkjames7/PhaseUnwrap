import numpy as np


def MatrixVariation(A):
	'''
	This function will attempt to define some measure of the variation 
	in the matrix.
	'''

	#measure 1 - just the sum of absolute differences between adjacent
	#matrix elements
	dx = A[1:] - A[:-1]
	dy = A[:,1:] - A[:,:-1]
	v1x = np.sum(np.abs(dx))
	v1y = np.sum(np.abs(dy))
	v1 = v1x + v1y
	
	#measure 2 - like above, but effectively a second derivative
	dx2 = dx[1:] - dx[:-1]
	dy2 = dy[:,1:] - dy[:,:-1]
	v2x = np.sum(np.abs(dx2))
	v2y = np.sum(np.abs(dy2))
	v2 = v2x + v2y	

	return v1,v2
