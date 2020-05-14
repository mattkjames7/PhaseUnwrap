import numpy as np
from ..Unwrapping.SortAngleDistribution import SortAngleDistribution

def DownSample(A,n):
	'''
	
	Inputs
	======
	A : numpy.ndarray
		2D array of either phases (angle distribution is minimized and 
		the mean of the angles used), or complex numbers (just use mean 
		of complex numbers)
	n : int or array-like
		1-element integer: defines the number of bins in both direction
		2-element array-like (list,tuple,ndarray): defines bins each direction
		
	'''
	
	#check if A is complex
	isC = np.iscomplex(A).any()
	
	#get input dims
	ii,ij = A.shape
	
	#output dims
	if np.size(n) == 1:
		oi,oj = n,n
	else:
		oi,oj = n
	 
	#create output array
	out = np.zeros((oi,oj),dtype=A.dtype)
	
	#calculate which bins each element belongs in
	Ii = np.int32(np.arange(ii)*(oi/ii))
	Ij = np.int32(np.arange(ij)*(oj/ij))
	
	Ij,Ii = np.meshgrid(Ij,Ii)
	
	#loop through each output bin
	for i in range(0,oi):
		uI = Ii == i
		for j in range(0,oj):
			use = np.where(uI & (Ij == j))
			tmp = A[use]
			if isC:
				out[i,j] = np.mean(tmp)
			else:
				s,_,_,_  = SortAngleDistribution(tmp,False,10)
				out[i,j] = np.mean(s)
	
	return out,Ii,Ij
