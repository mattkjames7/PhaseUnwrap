import numpy as np

def GetComplex(P,A):
	
	
	A2= A**2.0
	tP2 = np.tan(P)**2.0
	
	a2 = A2/(tP2 + 1)
	b2 = A2 - a2
	
	c = np.sqrt(a2) + np.sqrt(b2)*1.0j
	
	return c
