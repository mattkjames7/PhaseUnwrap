import numpy as np
import os

defargs = {	'VERBOSE':'TRUE',
			'LOGFILE':'snap.logfile',
			'TILECOSTTHRESH':500,
			'MINREGIONSIZE':100,
			'TILEEDGEWEIGHT':2.5,
			'SCNDRYARCFLOWMAX':8}
			
def Snaphu(C,**kwargs):
	
	
	
	#get the file names
	outfile = ''
	while outfile == '' or os.path.isfile(outfile):
		num = np.random.randint(1000000)
		outfile = '{:0d}.out'.format(num)
		
	infile = '{:0d}.in'.format(num)
	cfgfile = '{:0d}.cfg'.format(num)
	
	#save the config file
	f = open(cfgfile,'w')
	f.write('INFILE {:s}\n'.format(infile))
	f.write('LINELENGTH {:d}\n'.format(C.shape[1]))
	f.write('OUTFILE {:s}\n'.format(outfile))
	
	keys = list(defargs.keys())
	for k in keys:
		v = kwargs.get(k,defargs[k])
		if isinstance(v,str):
			o = k + ' {:s}\n'.format(v)
		elif isinstance(v,(np.int,np.int32,np.int64)):
			o = k + ' {:d}\n'.format(v)
		elif isinstance(v,(np.float,np.float32,np.float64)):
			o = k + ' {:f}\n'.format(v)
		f.write(o)
	f.close()
	
	#save the input data
	f = open(infile,'wb')
	C.flatten().astype('complex64').tofile(f)
	f.close()
	
	#run snaphu
	os.system('snaphu -f {:s}'.format(cfgfile))
	
	#remove input file
	os.system('rm -v {:s}'.format(infile))
	
	#read the output file
	f = open(outfile,'rb')
	c = np.fromfile(f,dtype='float32',count=C.size*2).reshape((C.shape[0]*2,C.shape[1]))
	f.close()
	
	#create output arrays of amplitude and phase
	ia = np.arange(C.shape[0])*2
	ip = np.arange(C.shape[0])*2 + 1
	
	A = c[ia]
	P = c[ip]
	
	#remove output file
	os.system('rm -v {:s}'.format(outfile))
	
	#remove config file
	os.system('rm -v {:s}'.format(cfgfile))
	
	return A,P
	
