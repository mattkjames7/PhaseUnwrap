import numpy as np
from .ReadMatrix import ReadMatrix

def ReadComplexMatrix(Name='-3-m-2'):
	'''
	Read one of the test matrices.
	
	'''
	A = ReadMatrix('A'+Name)
	P = ReadMatrix('P'+Name)
	C = A*np.exp(1j*P)
	return C
