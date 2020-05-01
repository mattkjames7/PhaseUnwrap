import numpy as np
from . import Globals
import PyFileIO as pf

def ReadMatrix(Name='P-1-m-2'):
	'''
	Read one of the test matrices.
	
	'''
	fname = Globals.MatrixPath + '{:s}.bin'
	f = open(fname,'rb')
	P = pf.ArrayFromFile(f,'float32')
	f.close()
	
	return P
