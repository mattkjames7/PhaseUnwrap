import numpy as np
from skimage.restoration import unwrap_phase

def Arevallilo2002(A):
	'''
	This routine uses the method from:
	
	Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and 
	Munther A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm 
	based on sorting by reliability following a noncontinuous path”, 
	Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002
	'''
	
	return unwrap_phase(A)
