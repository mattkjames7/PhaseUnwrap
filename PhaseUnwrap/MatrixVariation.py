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
	v1x = np.mean(np.abs(dx))
	v1y = np.mean(np.abs(dy))
	v1 = v1x + v1y
	
	#measure 2 - like above, but effectively a second derivative
	dx2 = dx[1:] - dx[:-1]
	dy2 = dy[:,1:] - dy[:,:-1]
	v2x = np.mean(np.abs(dx2))
	v2y = np.mean(np.abs(dy2))
	v2 = v2x + v2y	

	#measure 1 - just the sum of square differences between adjacent
	#matrix elements
	v3x = np.mean((dx)**2)
	v3y = np.mean((dy)**2)
	v3 = v3x + v3y
	
	#measure 2 - like above, but effectively a second derivative

	v4x = np.mean((dx2)**2)
	v4y = np.mean((dy2)**2)
	v4 = v4x + v4y	


	return np.array([v1,v2,v3,v4])
