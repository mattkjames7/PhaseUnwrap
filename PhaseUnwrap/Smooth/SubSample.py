import numpy as np
from ..Unwrapping.SortAngleDistribution import SortAngleDistribution


def SubSample(A,Window=3):
	'''
	Creates a smaller array using the mean of a bunch of square windows
	
	'''
	ni,nj = A.shape
	
	#calculate the size of the output 
	oi = ni//Window
	oj = nj//Window
	
	#create output array
	o = np.zeros((oi,oj),dtype='float32')
	
	for i in range(0,oi):
		i0 = i*Window
		i1 = (i+1)*Window
		for j in range(0,oj):
			j0 = j*Window
			j1 = (j+1)*Window
			print("\r{:6.2f}%".format(100.0*((i*oj) + j)/(oi*oj)),end="")

			tmp = A[i0:i1,j0:j1]
			srt,_,_,_  = SortAngleDistribution(tmp.flatten(),False,10)

			o[i,j] = np.mean(srt)
	print("\r{:6.2f}%".format(100.0*((i*oj) + j)/(oi*oj)))			
	return o
