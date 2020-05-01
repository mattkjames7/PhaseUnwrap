from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .ReadMatrix import ReadMatrix

def PlotMatrix3D(A,fig=None,maps=[1,1,0,0],scale=None,title=''):
	'''
	Plot the phase matrix in 3D
	'''

	#check if A is a string, if so load one of the test matrices
	if isinstance(A,str):
		A = ReadMatrix(A)

	#get array dimensions
	nx,ny = A.shape
	x = np.arange(nx)
	y = np.arange(ny)
	x,y = np.meshgrid(x,y)

	#colors
	norm = colors.Normalize()
	cmap = plt.cm.get_cmap('jet')

	#get the scale limits
	if scale is None:
		scale = [np.nanmin(A),np.nanmax(A)]


	#create the axes (and window if needed)
	if fig is None:
		fig = plt
		fig.figure()
	ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]),projection='3d')

	#plot the phase
	print(x.shape,y.shape,A.shape)
	sm = ax.plot_surface(x,y,A.T,norm=norm,vmin=scale[0],vmax=scale[1],cmap=cmap,linewidth=0.0)
	divider = make_axes_locatable(ax)
	cbar = fig.colorbar(sm,shrink=0.5, aspect=10) 
	ax.set_title(title)


	return ax
