import numpy as np


def SortAngleDistribution(Ain,Verbose=True,MaxLoops=np.inf):
	'''
	This will attempt to minimize the standard deviation of a bunch of
	angles (in radians) by adding a subtracting 2*pi
	'''
	
	if np.where(np.isfinite(Ain))[0].size <= 1:
		return Ain,0,0.0,0.0
	
	#copy the input array
	A = np.copy(Ain)
	
	#start by making everything within 0 - 2*pi
	A = A % (2*np.pi)
	 
	#calculate the initial stdev
	stdin = np.nanstd(A)
	
	#loop through each point until there are no changes
	n = A.size
	loops = 0
	changed = True
	st = np.zeros(3,dtype='float32')
	a = np.zeros(3,dtype='float32')
	dpi = np.array([-2*np.pi,0.0,2*np.pi])
	while changed and (loops < MaxLoops):
		std0 = np.nanstd(A)
		if Verbose:
			print('Loop {:d} std {:f}'.format(loops,std0))
		changed = False
		
		for i in range(0,n):
			if np.isfinite(A[i]):
				std0 = np.nanstd(A)
				st[1] = std0
				a[:] = dpi + A[i]
				#calculate stdev for point i + 2pi
				A[i] = a[2]
				st[2] = np.nanstd(A)
				#for - 2pi
				A[i] = a[0]
				st[0] = np.nanstd(A)
				
				#find the lowest
				ind = st.argmin()
				
				if ind == 1:
					#no change
					pass
				else:
					changed = True
				A[i] = a[ind]
			
		loops += 1
	stdout = np.nanstd(A)
	return A,loops,stdin,stdout
