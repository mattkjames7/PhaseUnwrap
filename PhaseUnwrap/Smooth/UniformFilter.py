import numpy as np
from scipy.ndimage import uniform_filter

def UniformFilter(A,Window=3):
	if np.iscomplex(A):
		r = uniform_filter(A.real,Window)
		i = uniform_filter(A.imag,Window)
		return r + 1j*i
	else:
		return uniform_filter(A,Window)
	
