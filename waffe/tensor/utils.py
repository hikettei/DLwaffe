
import pyopencl as cl
import pyopencl.array as clarr
import numpy as np


def sin(tensor):
	M = np.int32(tensor.dim()[0])
	N = np.int32(tensor.dim()[1])

	cpu_earr = np.empty((N, M), dtype=tensor.device.DTYPE)
	resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = Tensor(cpu_earr, device=tensor.device, x_buf=resbuff)

	event = tensor.device.prg.matsin(tensor.device.queue, (1,), None, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	return res

def log(tensor):
	M = np.int32(tensor.dim()[0])
	N = np.int32(tensor.dim()[1])

	cpu_earr = np.empty((N, M), dtype=tensor.device.DTYPE)
	resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
	res      = Tensor(cpu_earr, device=tensor.device, x_buf=resbuff)

	event = tensor.device.prg.matlog(tensor.device.queue, (1,), None, M, N, tensor.x_buf, res.x_buf)
	cl.wait_for_events([event, ])
	return res