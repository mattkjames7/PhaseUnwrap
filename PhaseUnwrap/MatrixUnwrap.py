import numpy as np
from .Unwrapping.Arevallilo2002 import Arevallilo2002
from .Smooth.DownSample import DownSample
from .Smooth.FFTSmooth import FFTSmooth
from .Smooth.UniformFilter import UniformFilter
from .PlotMatrix3D import PlotMatrix3D
import matplotlib.pyplot as plt


class MatrixUnwrap(object):
	def __init__(self,C):
		self.C0 = C
		self.P0 = np.angle(C)
		self.A0 = np.abs(C)
		
		self.P1 = None
		self.P2 = None
		self.dP = None
		self.Pu = None
		
	def Unwrap(self,DS=None,FFT=None,UF=None):
		
		#create a copy of the complex array
		C = np.copy(self.C0)
		
		#down sample
		if not DS is None:
			C,I,J = DownSample(C,DS)
			self.I = I
			self.J = J
			
		#FFT filter
		if not FFT is None:
			C = FFTSmooth(C,FFT)
			
		#Uniform filter
		if not UF is None:
			C = UniformFilter(C,UF)
			
		#extract phase before unwrapping
		self.P1 = np.angle(C)

		
		#unwrap
		self.P2 = Arevallilo2002(self.P1)
		
		
		#calculate change in phase
		self.dP = self.P2 - self.P1
		
		#if we down sampled then we need to adjust the original array
		if DS is None:
			self.Pu = self.P2
		else:
			dP = self.P2 - self.P1
			self.Pt = self.P1[I,J] + dP[I,J]
			self.Put = self.P0 + dP[I,J]
			dPt = 2*np.pi*np.round((self.Put - self.Pt)/(2*np.pi))
			self.Pu = self.Put - dPt
			
	def Plot(self):
		
		fig = plt
		fig.figure(figsize=(11,8))
		
		ax0 = PlotMatrix3D(self.A0,fig,[2,2,0,0],title='Amplitude')
		ax1 = PlotMatrix3D(self.P0,fig,[2,2,1,0],title='Wrapped Phase')
		ax2 = PlotMatrix3D(self.P2,fig,[2,2,0,1],title='Unwrapped Phase')
		ax3 = PlotMatrix3D(self.Pu,fig,[2,2,1,1],title='Unwrapped Phase')
