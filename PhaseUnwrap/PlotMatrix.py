import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .ReadMatrix import ReadMatrix

def PlotMatrix(A,fig=None,maps=[1,1,0,0],scale=None,title=''):
	'''
	Plot the phase matrix in 2D
	'''

	#check if A is a string, if so load one of the test matrices
	if isinstance(A,str):
		A = ReadMatrix(A)

	#colors
	norm = colors.Normalize()
	cmap = plt.cmap.get_cmap('jet')

	#get the scale limits
	if scale is None:
		scaleg = [np.nanmin(A),np.nanmax(A)]
	else:
		scaleg = scale


	#create the axes (and window if needed)
	if fig is None:
		fig = plt
		fig.figure()
	ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))

	#plot the phase
	sm = ax.imshow(A,norm=norm,vmin=scale[0],vmax=scale[1],cmap=cmap,aspect='auto')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = fig.colorbar(sm,cax=cax) 
	ax.set_title(title)


	return ax
