import numpy as np


def FFTSmooth(Ain,keep=0.2):
	'''
	Uses a 2D FFT of the phases to remove short wavelength variations.
	
	'''
	
	F = np.fft.fft2(Ain)
	ni,nj = F.shape
	F[int(ni*keep):int(ni*(1-keep))] = 0
	F[:,int(nj*keep):int(nj*(1-keep))] = 0
	A = np.fft.ifft2(F)
	
	return A.real
