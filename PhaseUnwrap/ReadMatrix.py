import numpy as np
from . import Globals
import PyFileIO as pf

def ReadMatrix(Name='P-3-m-2'):
	'''
	Read one of the test matrices.
	
	'''
	fname = Globals.MatrixPath + '{:s}.bin'.format(Name)
	f = open(fname,'rb')
	P = pf.ArrayFromFile('float32',f)
	f.close()
	
	return P
