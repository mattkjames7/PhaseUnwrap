import numpy as np
import copy

def _GetIntervalIDs(A):
	'''
	This will split the data into 6 intervals between -pi and +pi, 
	returning an integer array the same shape as A. The integer of a
	given element corresponds to the interval bin.
	'''
	print("Assigning interval IDs")
	I = np.int32(np.round((6.0*A)/(2.0*np.pi)))
	return I
	
def _ExcludeBridges(Iin):
	'''
	This bit should hopefully remove the interval IDs of the small
	"bridges" which narrowly join larger regions. In the paper it says
	that the 1st, 2nd and 3rd neighbours of a point must not have the 
	same ID in at least 2 of the 3 directions (x, y and z). As this is
	a 2D algorithm, I will assum that this has to be 1 out of 2 dimensions.
	'''
	print("Excluding bridges")
	#create output array
	I = copy.deepcopy(Iin)
	
	#get array shape
	ni,nj = I.shape
	
	#get the limits
	I0 = (np.arange(ni) - 3).clip(min=0)
	I1 = (np.arange(ni) + 4).clip(max=ni)
	J0 = (np.arange(nj) - 3).clip(min=0)
	J1 = (np.arange(nj) + 4).clip(max=nj)
	
	bad = 0
	for i in range(0,ni):
		i0 = I0[i]
		i1 = I1[i]

		for j in range(0,nj):
			j0 = J0[j]
			j1 = J1[j]


			print("\r{:6.2f}% - Bad: {:d}".format(100.0*((i*nj) + j)/(ni*nj),bad),end="")
			if Iin[i,j] != -1:	
				#check x direction
				badx = (np.sum(Iin[i0:i1,j] == Iin[i,j]) <= 1)
				
				#check y direction
				bady = (np.sum(Iin[i,j0:j1] == Iin[i,j]) <= 1)
				
				if badx or bady:
					I[i,j] = -1
					bad += 1
	print("\r{:6.2f}% - Bad: {:d}".format(100.0*((i*nj) + j)/(ni*nj),bad))
	return I,bad
	
def _ExpandRegion(i,j,r,R,I):
	'''
	This function will test each of the four neighbours of a point to see
	if it has the same interval integer, then it will call itself recursively
	'''
	
	#check the bin has not already been assigned
	if R[i,j] > -1:
		return R
		
	#Assign current region integer to this point
	R[i,j] = r
	
	#check if left neighbour matches
	if i > 0:
		if I[i,j] == I[i-1,j]:
			R = _ExpandRegion(i-1,j,r,R,I)
	
	#check right neighbour
	if i < I.shape[0]-1:
		if I[i,j] == I[i+1,j]:
			R = _ExpandRegion(i+1,j,r,R,I)	

	#check if bottom neighbour matches
	if j > 0:
		if I[i,j] == I[i,j-1]:
			R = _ExpandRegion(i,j-1,r,R,I)
	
	#check top neighbour
	if j < I.shape[1]-1:
		if I[i,j] == I[i,j+1]:
			R = _ExpandRegion(i,j+1,r,R,I)	

	return R
	

			
			
	
def _GetRegions(I):
	'''
	This will attempt to determine the regions of each pixel by looking
	at nearest neighbours.
	
	'''
	print("Labeling regions")
	#because this uses a recursive function, need to set the recursion limit
	import sys
	sys.setrecursionlimit(100000)
	
	ni,nj = I.shape
	
	#output region array, fill with -1 to identifiy that the pixel has
	# not been assigned yet
	R = np.zeros((ni,nj),dtype='int32') - 1
	
	#loop through each element
	r = 0
	for i in range(0,ni):
		for j in range(0,nj):
			print("\r{:6.2f}% - Regions: {:d}".format(100.0*((i*nj) + j)/(ni*nj),r+1),end="")
			if R[i,j] == -1 and I[i,j] > -1:
				#recursively fill regions
				R = _ExpandRegion(i,j,r,R,I)
				r += 1
	print("\r{:6.2f}% - Regions: {:d}".format(100.0*((i*nj) + j)/(ni*nj),r+1))
	return R

def _RegionStats(R):
	'''
	This will list all of the regions, their sizes and the length of
	their borders
	
	'''
	print("Counting Region Borders")
	#regions
	Ri,Rs = np.unique(R[R > -1],return_counts=True)
	
	#now for the borders
	ni,nj = R.shape
	Rb = np.zeros(Ri.size,dtype='int32')
	for i in range(0,ni):
		for j in range(0,nj):
			print("\r{:6.2f}%".format(100.0*((i*nj) + j)/(ni*nj)),end="")
			#check if this point belongs to a region
			if R[i,j] > -1:
				c = 0
				#count left side
				if i > 0:
					c += np.int32(R[i,j] != R[i-1,j])
				#right side
				if i < ni-1:
					c += np.int32(R[i,j] != R[i+1,j])
				#bottom side
				if j > 0:
					c += np.int32(R[i,j] != R[i,j-1])
				#top side
				if j < nj-1:
					c += np.int32(R[i,j] != R[i,j+1])
				Rb[R[i,j]] += c
	print("\r{:6.2f}%".format(100.0*((i*nj) + j)/(ni*nj)))
	return Ri,Rs,Rb
		
def _NearestRegion(i,j,size,R,Ri,Rs):
	'''
	Scan a square for the nearest region
	
	'''
	#get the temporary region
	ni,nj = R.shape
	i0 = np.max([i-size,0])
	i1 = np.min([i+size+1,ni])
	j0 = np.max([j-size,0])
	j1 = np.min([j+size+1,nj])
	
	Rtmp = R[i0:i1,j0:j1]
	
	#check if there are any regions in there
	Rgood = Rtmp > -1
	
	if not Rgood.any():
		#if there are none then return -1
		return -1
	
	#if there are, find their numbers
	Ru = np.unique(Rtmp[Rgood])
	
	if Ru.size == 1:
		#simple, only one neighbouring group
		return Ru[0]
	else:
		#in this case, we have to find the biggest one
		sizes = np.zeros(Ru.size)
		
		for i in range(0,Ru.size):
			use = np.where(Ri == Ru[i])[0][0]
			sizes = Rs[use]
		
		use = np.where(sizes == np.max(sizes))[0][0]
		return Ru[use]
		
	
				
def _AssignExcluded(Rin,Ri,Rs):
	'''
	This will loop through all of the excluded points to try and assign 
	them to their nearest group - if multiple groups are in the same
	distance, then the largest is selected
	
	'''		
	print("Assigning region ID to excluded points")
	#create output array
	R = copy.deepcopy(Rin)
	
	#locate them
	E = np.where(R == -1)
	nE = E[0].size
	
	#loop through each one
	ni,nj = R.shape
	for e in range(0,nE):
		print("\r{:6.2f}%".format(100.0*(e + 1)/nE),end="")
		i,j = E[0][e],E[1][e]
		
		#search for regions within a square around the current point,
		#each time this fails - the search distance is increased by 1
		#square width = 2*size + 1
		size = 1
		done = False
		while not done and size < ni//2 and size < nj//2:
			r = _NearestRegion(i,j,size,R,Ri,Rs)
			if r > -1:
				R[i,j] = r
				done = True
			else:
				size += 1
	
	print("\r{:6.2f}%".format(100.0*(e + 1)/nE))
	return R
		

def _FindNeighbours(r,R):
	'''
	This will search for all neighbours of a given group r.
	'''
	#list all of the indices of points within the group r
	use = np.where(R == r)
	nu = use[0].size
	
	#create output array
	uR = np.unique(R)
	nR = uR.size
	N = np.zeros(nR,dtype='int32') - 1
	
	#loop through each point in r
	nN = 0
	ni,nj = R.shape
	for k in range(0,nu):
		print("\r{:6.2f}%".format(100.0*(k + 1)/nu),end="")
		i,j = use[0][k],use[1][k]
		
		#check left
		if i > 0:
			if R[i-1,j] != R[i,j] and not R[i-1,j] in N:
				N[nN] = R[i-1,j]
				nN += 1
		#check right
		if i < ni-1:
			if R[i+1,j] != R[i,j] and not R[i+1,j] in N:
				N[nN] = R[i+1,j]
				nN += 1
		#check bottom
		if j > 0:
			if R[i,j-1] != R[i,j] and not R[i,j-1] in N:
				N[nN] = R[i,j-1]
				nN += 1
		#check top
		if j < nj-1:
			if R[i,j+1] != R[i,j] and not R[i,j+1] in N:
				N[nN] = R[i,j+1]
				nN += 1
	print("\r{:6.2f}%".format(100.0*(k + 1)/nu))	
		
	return N[:nN]
	
def _NeighbourInterfaceIndices(r,n,R):
	'''
	List all of the interfaces between the main region and its neighbour
	'''
	dtype = [	('r0','int32',(2,)),	#position in r at edge
				('r1','int32',(2,)),	#position of voxel behind r0
				('n0','int32',(2,)),]	#position of neighbouring voxel
	
	#find left interfaces (n left of r)
	left0 = np.where((R[1:-1,:] == r) & (R[2:,:] == r) & (R[:-2,:] == n))
	left0 = np.where((R[1:-1,:] == r) & (R[2:,:] == r) & (R[:-2,:] == n))
	
	
	
def Karsa2019(Ain):
	'''
	This code is based on the SEGUE algorithm described in 10.1109/TMI.2018.2884093
	
	Because I am thick - this is only a 2D version of the algorithm, and
	it may not be exactly right in places.
	'''
	
	#Partitioning
	#0. Make sure that all values are between [0.0, 2*np.pi)
	A = Ain % (2.0*np.pi)
	
	#1. Split data into  6 intervals
	I0 = _GetIntervalIDs(A)
	
	#2. Exclude bridges (not sure if you're meant to repeatedly do this, or just once)
	bad = 1
	I = copy.deepcopy(I0)
	while bad > 0:
		I,bad = _ExcludeBridges(I)
	
	#3. Group regions
	R = _GetRegions(I)
	
	#4. Get region info, their sizes and borders
	Ri,Rs,Rb = _RegionStats(R)
	
	#5. Reassign Excluded
	R = _AssignExcluded(R,Ri,Rs)	
	
	#6. Update the region info to include the excluded points
	Ri,Rs,Rb = _RegionStats(R)
	Rdone = np.zeros(Ri.size,dtype='bool')
	
	#Merging!
	#set Plimit values
	Plimit = [0.3,0.1,0.0]
	
	#loop through each one
	for i in range(0,3):
		Plim = Plimit[i]
	
		
		#stop the first loop when most of the regions are unwrapped
		cont = True
		while cont:
			nochange = True
			
			#1. Start with the one with the largest border
			Ind = Rb.argmax()
			Rdone[Ind] = True
			
			#2. Look for all neighbouring regions
			N = _FindNeigbours(Ri[Ind],R)



			#Check the total number of 
