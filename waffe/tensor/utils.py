
import waffe as wf
import pyopencl as cl
import pyopencl.array as clarr
import numpy as np

# マクロ使いてぇ〜〜〜

def create_res_buffer(tensor):
	M = np.int32(tensor.dim()[0])
	N = np.int32(tensor.dim()[1])

	cpu_earr = np.empty((M, N), dtype=tensor.device.DTYPE)
	resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=tensor.device, x_buf=resbuff, extend=tensor, is_constant=False)

	return (int(M), int(N)), (int(tensor.device.TS), int(tensor.device.TS)), M, N, res

def sin(tensor, require_grad=True):
	def sin_backward(tensor):
		def _sin_backward(x):
			return wf.cos(tensor, require_grad=False)
		return _sin_backward

	gsize, lsize, M, N, res = create_res_buffer(tensor)

	event = tensor.device.prg.matsin(tensor.device.queue, gsize, lsize, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	if require_grad:
		wf.register_derivative(res, sin_backward(tensor), tensor)
		wf.register_variables(res, [tensor])
	res.sync()
	return res

def cos(tensor, require_grad=True):
	def cos_backward(tensor):
		def _cos_backward(x):
			return wf.sin(tensor, require_grad=False) * -1
		return _cos_backward

	gsize, lsize, M, N, res = create_res_buffer(tensor)

	event = tensor.device.prg.matcos(tensor.device.queue, gsize, lsize, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])

	if require_grad:
		wf.register_derivative(res, cos_backward(tensor), tensor)
		wf.register_variables(res, [tensor])
	res.sync()
	return res

def log(tensor, require_grad=True):
	def log_backward(tensor):
		def _log_backward(x):
			return 1 / tensor.detach()
		return _log_backward

	gsize, lsize, M, N, res = create_res_buffer(tensor)

	event = tensor.device.prg.matlog(tensor.device.queue, gsize, lsize, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])

	if require_grad:
		wf.register_derivative(res, log_backward(tensor), tensor)
		wf.register_variables(res, [tensor])
	res.sync()

	return res

def sigmoid(tensor, require_grad=True):
	def sigmoid_backward(tensor):
		def _sigmoid_backward(x):
			gsize, lsize, M, N, res = create_res_buffer(tensor)
			event = tensor.device.prg.dsigmoid(tensor.device.queue, gsize, lsize, M, N, tensor.x_buf, res.x_buf)
			cl.wait_for_events([event, ])
			return res
		return _sigmoid_backward

	gsize, lsize, M, N, res = create_res_buffer(tensor)

	event = tensor.device.prg.sigmoid(tensor.device.queue, gsize, lsize, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])

	if require_grad:
		wf.register_derivative(res, sigmoid_backward(tensor), tensor)
		wf.register_variables(res, tensor.variables)
	res.sync()

	return res

