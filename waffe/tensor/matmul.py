
import waffe as wf
import numpy as np

import pyopencl as cl
import pyopencl.array as clarr


def _DotProductBackward(L, X, Y):
	def DotProductBackward(g):
		if g == X:
			return (X @ L / (X.transpose() @ X)).no_grad()
		elif g == Y:
			return (Y @ L / (Y.transpose() @ Y)).no_grad()
		else:
			Exception("")

	return DotProductBackward

def _dot_product(A, B):
	assert A.dim()[0] == A.dim()[0], ""
	M = np.int32(A.dim()[0])
	N = np.int32(A.dim()[1])

	cpu_earr = np.empty((M, N), dtype=A.device.DTYPE)
	resbuff  = cl.Buffer(A.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=A.device, x_buf=resbuff, extend=A, is_constant=False)

	gsize = (int(M), int(N))
	lsize = (int(A.device.TS), int(A.device.TS))

	event = A.device.prg.matdot(A.device.queue, gsize, lsize, M, N, A.x_buf, B.x_buf, res.x_buf)
	cl.wait_for_events([event, ])

	wf.register_derivative(res, _DotProductBackward(res, A, B), A, B)
	wf.register_variables(res, [A, B])
	return res

def _matrix_vector_product(A, B):
	"""
	Inputs : A ... 2darray
			 B ... 1darray
	"""
	return _matrix_matrix_product(A, B).expand_dims(0)

def _MMProductBackward(L, X, Y):
	def MMProductBackward(g):
		if g == X:
			if (g.grad.detach() == [[1.]]).any():
				return X.transpose()
			else:
				return g.grad @ X.transpose()
		elif g == Y:
			if (g.grad.detach() == [[1.]]).any():
				return Y.transpose()
			else:
				return Y.transpose() @ g.grad
		else:
			Exception("")
	return MMProductBackward

def _matrix_matrix_product(A, B):
	#2x2
	#assert A.dim()[1] == B.dim()[0], "cannot multiply mat1 and mat2 due to mismatch of A.dim()[0]"
	M = np.int32(A.dim()[0])
	K = np.int32(A.dim()[0])
	N = np.int32(B.dim()[0])

	cpu_earr = np.empty((M, N), dtype=A.device.DTYPE)
	resbuff  = cl.Buffer(A.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=A.device, x_buf=resbuff, extend=A, is_constant=False)

	gsize = (int(M/A.device.WPTM), int(N/A.device.WPTM))
	lsize = (int(A.device.TSM/A.device.WPTM), int(A.device.TSM/A.device.WPTM))
	event = A.device.prg.mm_product(A.device.queue, gsize, lsize, M, N, K, A.x_buf, B.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	
	wf.register_derivative(res, _MMProductBackward(res, A, B), A, B, deep_variables=False)
	wf.register_variables(res, [A, B])
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

