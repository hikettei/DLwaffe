
import waffe as wf
import numpy as np

import pyopencl as cl
import pyopencl.array as clarr

def MatMulBackward():
	pass

def _dot_product(A, B):
	pass

def _matrix_vector_product(A, B):
	pass

def _matrix_matrix_product(A, B):
	#2x2
	assert A.dim()[0] == B.dim()[1], "cannot multiply mat1 and mat2 due to mismatch of A.dim()[0]"
	M = np.int32(A.dim()[1])
	K = np.int32(A.dim()[0])
	N = np.int32(B.dim()[1])

	cpu_earr = np.empty((N, M), dtype=A.device.DTYPE)
	resbuff  = cl.Buffer(A.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=A.device, x_buf=resbuff, extend=A, is_constant=False)

	gsize = (int(M/A.device.WPTM), int(N/A.device.WPTM))
	lsize = (int(A.device.TSM/A.device.WPTM), int(A.device.TSM/A.device.WPTM))
	event = A.device.prg.mm_product(A.device.queue, gsize, lsize, M, N, K, A.x_buf, B.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	# matbackward
	return res

def _batch_product(A, B):
	pass

def matmul(A, B):
	""" Computes the matrix multiplication of two arrays

	Inputs:
		A (class wf.Tensor) The left operand of the matrix multiplication.
		B (class wf.Tensor) The right operand of the matrix multiplication.

	Returns:
		1. if 'a' and 'b' are both 1d array, the dot product of 'a' and 'b' is returned.
		2. if both are 2d array, the matrix-matrix product is returned.
		3. if A is 1d array and B is 2d array,  ...?
		4. if A is 2d array and B is 1d array, the matrix-vector product is returned.
		5. if either dimension is larget than 3, batch filter is adapted.
	"""

	if len(A.d_shape) == 1 and len(B.d_shape) == 1:
		# <=> 1. dot_product
		return _dot_product(A, B)

	elif len(A.d_shape) == 2 and len(B.d_shape) == 2:
		# <=> 2, matrix_matrix_product
		return _matrix_matrix_product(A, B)

	elif len(A.d_shape) == 1 and len(B.d_shape) == 2:
		# <=> 3.
		return _matrix_vector_product(B, A) # ???

	elif len(A.d_shape) == 2 and len(B.d_shape) == 1:
		# <=> 4.
		return _matrix_vector_product(A, B)

	else:
		# dim is larger than 3
		return _batch_product(A, B)