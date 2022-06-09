
import waffe as wf
import pyopencl as cl
import pyopencl.array as clarr
import numpy as np

#@wf.tensor_util(tensor)
def sin(tensor, require_grad=True):
	M = np.int32(tensor.dim()[0])
	N = np.int32(tensor.dim()[1])

	cpu_earr = np.empty((N, M), dtype=tensor.device.DTYPE)
	resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=tensor.device, x_buf=resbuff, extend=tensor)

	event = tensor.device.prg.matsin(tensor.device.queue, (1,), None, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	res.sync()

	if require_grad:
		wf.register_backwards_value(res, lambda y: tensor/y * wf.cos(tensor, require_grad=False))
		wf.register_backwards_node(res, var=tensor.variables)
	return res

def cos(tensor, require_grad=True):
	M = np.int32(tensor.dim()[0])
	N = np.int32(tensor.dim()[1])

	cpu_earr = np.empty((N, M), dtype=tensor.device.DTYPE)
	resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=tensor.device, x_buf=resbuff, extend=tensor)

	event = tensor.device.prg.matcos(tensor.device.queue, (1,), None, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	res.sync()

	if require_grad:
		wf.register_backwards_value(res, lambda y: tensor/y * -wf.sin(tensor, require_grad=False))
		wf.register_backwards_node(res, tensor)
	return res

def log(tensor, require_grad=True):
	M = np.int32(tensor.dim()[0])
	N = np.int32(tensor.dim()[1])

	cpu_earr = np.empty((N, M), dtype=tensor.device.DTYPE)
	resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = wf.Tensor(cpu_earr, device=tensor.device, x_buf=resbuff, extend=tensor)

	event = tensor.device.prg.matlog(tensor.device.queue, (1,), None, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	res.sync()

	if require_grad:
		wf.register_backwards_value(res, lambda y: y * 1/wf.log(tensor/y, require_grad=False).detach())
		wf.register_backwards_node(res, tensor)
	return res