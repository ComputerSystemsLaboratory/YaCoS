import numpy as np
cimport numpy as cnp
import time
from libc.math cimport round

cdef double get_coord(int max_coord, double coord) nogil:
	cdef double r 
	r=round(coord)
	if (r>=0) and (r < max_coord):
		return r
	return -1

cpdef cnp.ndarray cython_lbpeq(cnp.ndarray arg_img, int P=8, int R=2):
	'''
		This function extracts the binary pattern around each "pixel of
		the binary image. The strategy is very similar to LBP, but
		instead of compare the pixel with their neighbors, it just take
		the pattern looking at its neighbors.
		Exemple:
		Supose this part of an BINARY image and the pixel X. 
		0   1   1
		1	X	1			
		0	0	1

		Consider P=8 and R=1

		Pattern around X is 11001011.

		>-------V
		:		:
		: pixel	: <---pattern start here
		:		:
		^-------<


	'''
	cdef double[:,::1] img = np.ascontiguousarray(arg_img,dtype=np.double)
	cdef int totlin = arg_img.shape[0]
	cdef int totcol = arg_img.shape[1] 
	cdef int r = 0
	cdef int c = 0
	cdef int p = 0
	cdef double pattern = 0
	cdef double m = 0
	cdef double pixel

	cdef cnp.ndarray rr = -R*np.sin(2*np.pi*np.arange(P,dtype=np.double)/P)
	cdef cnp.ndarray cc = R*np.cos(2*np.pi*np.arange(P,dtype=np.double)/P)
	cdef double[:, ::1] rbp_matrix = np.zeros([totlin,totcol],dtype=np.double)
	cdef double[::1] rp = np.round(rr, 5)
	cdef double[::1] cp = np.round(cc, 5)
	cdef int x,y
	while r < totlin:
		c=0
		while c < totcol:
			pattern = 0
			p=0
			m=1
			while p < P:
				x = <int>get_coord(totlin,r+rp[p])
				y = <int>get_coord(totcol,c+cp[p])
				if x != -1 and y != -1:
					pixel = img[x,y]
					pattern += pixel * m
				m = m*2
				p = p+1
			rbp_matrix[r,c]=pattern
			c+=1
		r+=1

	return np.asarray(rbp_matrix)
