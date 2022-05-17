import numpy as np
from ..Unwrapping.SortAngleDistribution import SortAngleDistribution

def PhaseSmooth(Ain,Window=1,Function='mean'):
	'''
	An attempt to smooth phase using a square window
	
	In a window, phases will be unwrapped to be as close as possible
	
	Then the central value will be replaced by some function of all
	the points within the window.
	
	'''
	#get the array dimenstions
	ni,nj = Ain.shape
	
	#create the output array
	A = np.zeros(Ain.shape,dtype='float32')

	I0 = (np.arange(ni) - Window).clip(min=0)
	I1 = (np.arange(ni) + Window + 1).clip(max=ni)
	J0 = (np.arange(nj) - Window).clip(min=0)
	J1 = (np.arange(nj) + Window + 1).clip(max=nj)
	
	for i in range(0,ni):
		i0 = I0[i]
		i1 = I1[i]
		for j in range(0,nj):
			j0 = J0[j]
			j1 = J1[j]
			print("\rSmoothing {:6.2f}%".format(100.0*((i*nj) + j)/(ni*nj)),end="")

			tmp = Ain[i0:i1,j0:j1]
			srt,_,_,_  = SortAngleDistribution(tmp.flatten(),False,10)
			srt = srt.reshape(tmp.shape)

			A[i,j] = np.mean(srt)


	print("\rSmoothing {:6.2f}%".format(100.0*((i*nj) + j)/(ni*nj)))
	return A
	
