import numpy as np
from ..Smooth.SubSample import SubSample
import copy
from ..MatrixVariation import MatrixVariation

class RandomChange(object):
	def __init__(self,Ain,Window=0):
		if Window > 0:
			self.A0 = SubSample(Ain,Window)
		else:
			self.A0 = copy.deepcopy(Ain)
			
		self.A = copy.deepcopy(self.A0)
		self.v = None
		
	def Train(self,nEpoch=1000,Pgood=0.99,Pbad=0.0001,vI=0):
		
		ni,nj = self.A0.shape		
		A = copy.deepcopy(self.A0)
	
	
		v = np.zeros((nEpoch,4),dtype='float32')		
		vp = MatrixVariation(A)	
		if self.v is None:
			self.v = np.zeros((1,4),dtype='float32')
			self.v[0] = vp
			step0 = 1
		else:
			step0 = self.v.shape[0]
		
		for e in range(0,nEpoch):
			dxy = np.pi
			if np.random.rand() < 0.5:
				dx = np.abs(A[1:] - A[:-1])
				badx = np.where(dx > dxy)
				while badx[0].size < 10:
					dxy*=0.9
					badx = np.where(dx > dxy)
				I = np.random.randint(0,badx[0].size)
				i = badx[0][I] + np.random.randint(0,2)
				j = badx[1][I]
			else:
				dy = np.abs(A[:,1:] - A[:,:-1])
				bady = np.where(dy > dxy)
				while bady[0].size < 10:
					dxy*=0.9
					bady = np.where(dy > dxy)				
				I = np.random.randint(0,bady[0].size)
				i = bady[0][I]
				j = bady[1][I] + np.random.randint(0,2)
			
			#i_ = np.array([i-1,i+1]).clip(min=0,max=ni-1)
			#j_ = np.array([j-1,j+1]).clip(min=0,max=nj-1)
			
			#p = np.array([A[i_[0],j],A[i_[1],j],A[i,j_[0]],A[i,j_[1]]])
			#diff = np.int32(np.round((p - A[i,j])/(2*np.pi)))
			#u,c = np.unique(diff,return_counts=True)

			
			#print('\n',i,j,A[i,j] - p,diff,'\n')
			#diff = 2*np.pi*(u[c.argmax()])
			diff = 2*np.pi*((np.random.rand()-0.5 >= 0)-0.5)*2
			A[i,j] += diff

			vt = MatrixVariation(A)
			
			dv = vt[vI] - vp[vI]
			print('\n',dv,'\n')
			if dv < 0:
				if np.random.rand() <= Pgood:
					keep = True
				else:
					keep = False
			else:
				if np.random.rand() <= Pbad:
					keep = True
				else:
					keep = False
	
			if keep:
				vp = copy.deepcopy(vt)
				
			else:
				A[i,j] -= diff
		
			v[e] = vp
			
			print('\rEpoch {0} of {1}; Cost: {2}'.format(e+1,nEpoch,vp[vI]),end='')
		print()
		self.v = np.concatenate((self.v,v),axis=0)
		self.A = A


	def Train2(self,nEpoch=1000,Pkeep=0.9,vI=0):
		
		ni,nj = self.A0.shape		
		A = copy.deepcopy(self.A0)
	
	
		v = np.zeros((nEpoch,4),dtype='float32')		
		vp = MatrixVariation(A)	
		if self.v is None:
			self.v = np.zeros((1,4),dtype='float32')
			self.v[0] = vp
			step0 = 1
		else:
			step0 = self.v.shape[0]
		
		for e in range(0,nEpoch):
			
	
			dxy = np.pi
			if np.random.rand() < 0.5:
				dx = np.abs(A[1:] - A[:-1])
				badx = np.where(dx > dxy)
				while badx[0].size < 10:
					dxy*=0.9
					badx = np.where(dx > dxy)
				I = np.random.randint(0,badx[0].size)
				i = badx[0][I] + np.random.randint(0,2)
				j = badx[1][I]
			else:
				dy = np.abs(A[:,1:] - A[:,:-1])
				bady = np.where(dy > dxy)
				while bady[0].size < 10:
					dxy*=0.9
					bady = np.where(dy > dxy)				
				I = np.random.randint(0,bady[0].size)
				i = bady[0][I]
				j = bady[1][I] + np.random.randint(0,2)
			
			i_ = np.array([i-1,i+1]).clip(min=0,max=ni-1)
			j_ = np.array([j-1,j+1]).clip(min=0,max=nj-1)
			
			p = np.array([A[i_[0],j],A[i_[1],j],A[i,j_[0]],A[i,j_[1]]])
			diff = np.int32(np.round((p - A[i,j])/(2*np.pi)))
			u,c = np.unique(diff,return_counts=True)

			
			#print('\n',i,j,A[i,j] - p,diff,'\n')
			diff = 2*np.pi*(u[c.argmax()])
			A[i,j] += diff

			vt = MatrixVariation(A)
			
			dv = vt[vI] - vp[vI]
			print('\n',dv,'\n')
			if dv < 0:
				if np.random.rand() <= Pkeep:
					keep = True
				else:
					keep = False
			else:
				if np.random.rand() <= 0.01:
					keep = True
				else:
					keep = False
	
			if keep:
				vp = copy.deepcopy(vt)
				
			else:
				A[i,j] -= diff
		
			v[e] = vp
			
			print('\rEpoch {0} of {1}; Cost: {2}'.format(e+1,nEpoch,vp[vI]),end='')
		print()
		self.v = np.concatenate((self.v,v),axis=0)
		self.A = A

