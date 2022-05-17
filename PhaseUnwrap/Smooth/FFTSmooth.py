import numpy as np


def FFTSmooth(Ain,keep=0.2):
	'''
	Uses a 2D FFT of the phases to remove short wavelength variations.
	
	'''
	
	if np.iscomplex(Ain).any():
	
		Fr = np.fft.fft2(Ain.real)
		Fi = np.fft.fft2(Ain.imag)
		ni,nj = Fr.shape
		Fr[int(ni*keep):int(ni*(1-keep))] = 0
		Fr[:,int(nj*keep):int(nj*(1-keep))] = 0
		Fi[int(ni*keep):int(ni*(1-keep))] = 0
		Fi[:,int(nj*keep):int(nj*(1-keep))] = 0
		Ar = np.fft.ifft2(Fr)
		Ai = np.fft.ifft2(Fi)
		
		out = Ar + 1j*Ai
	
	else:
		F = np.fft.fft2(Ain)
		ni,nj = F.shape
		F[int(ni*keep):int(ni*(1-keep))] = 0
		F[:,int(nj*keep):int(nj*(1-keep))] = 0
		A = np.fft.ifft2(F)
	
		out = A.real

	return out
