
import waffe as wf
from .tensor import utils

from .str_format import wftensor_to_str
import pyopencl as cl
import pyopencl.array as clarr

from .kernel import MAT_KERNELS

import numpy as np
from enum import Enum

DTYPES = {np.float16 : 'half', np.float32 : 'float', np.float64 : 'double'}

class WaffeDevice():
    def __init__(self, cl_device, **kwargs):
        self.DTYPE = kwargs['DTYPE'] if 'DTYPE' in kwargs else np.float32

        assert self.DTYPE in DTYPES

        floatX = DTYPES[self.DTYPE]
        self.TSM = kwargs['TSM'] if 'TSM' in kwargs else 128
        self.TSN = kwargs['TSN'] if 'TSN' in kwargs else 128
        self.TSK = kwargs['TSK'] if 'TSK' in kwargs else 8
        self.TS = kwargs['TS'] if 'TS' in kwargs else 16
        self.WPTM = kwargs['WPTM'] if 'WPTM' in kwargs else 8
        self.WPTN = kwargs['WPTN'] if 'WPTN' in kwargs else 8

        self.cl_device = cl_device
        self.ctx      = cl.Context([cl_device])
        self.queue    = cl.CommandQueue(self.ctx)

        options = "-DTSM={} -DTSN={} -DTSK={} -DWPTM={} -DWPTN={} -DfloatX={} -DTS={} -cl-mad-enable -cl-fast-relaxed-math".format(
            self.TSM, self.TSN, self.TSK, self.WPTM, self.WPTN, floatX, self.TS)

        self.prg = cl.Program(self.ctx, MAT_KERNELS).build(options)

class Tensor():
    def __init__(self, x, dtype=None, device=None, x_buf=None):
        """
        Exa: wf.Tensor([1,2,3])
        Arguments:
            x ... the type of x is as follows: list, numpy.array, numpy.ndarray
        """

        assert isinstance(x, list) or type(x).__module__ == np.__name__, "The first argument of wf.Tensor must be list or numpy list"        

        if isinstance(x, list):
            x = np.asarray(x).astype(np.float32)

        if dtype is None:
            dtype = x.dtype

        if device is None:
            device = wf.get_device("device:0") # In default, use cpu

        # Restore device, dtype
        self.device = device
        self.dtype  = dtype

        self.x     = x #remove it in the future, because it makes memory-usage 2*n
        if x_buf is None:
            self.x_buf = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.x)
        else:
            self.x_buf = x_buf

    def __str__(self):
        return wftensor_to_str(self.x)

    def __matmul__(self, y):
        assert self.dim()[0] == y.dim()[0], "The mismatch shape"

        M = np.int32(self.dim()[0])
        K = np.int32(self.dim()[0])
        N = np.int32(y.dim()[1])

        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff  = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res      = Tensor(cpu_earr, device=self.device, x_buf=resbuff)

        #global_sizes = (int(M/self.device.WPTM), int(N/self.device.WPTN))
        #local_sizes = (int(self.device.TSM/self.device.WPTM), int(self.device.TSN/self.device.WPTN))
        #print(global_sizes)
        #print(local_sizes)
        event = self.device.prg.matmul(self.device.queue, (int(M/self.device.WPTM),), None, M, N, K, self.x_buf, self.x_buf, res.x_buf)
        cl.wait_for_events([event, ])
        
        return res

    def to_list(self):
        # x.to_list() returns list
        return self.x

    def reshape_self(self, dim):
        # it is destructive ver
        self.x = self.x.reshape(dim)
        return self

    def reshape(self, dim):
        # it is not destructive
        return Tensor(self.x.reshape(dim), dtype=self.dtype, device=self.device)

    def transpose(self, axes=None):
        #utils.transpose(self, axes=axes)
        return self

    def dim(self):
        return self.x.shape


def empty(dim, dtype=None, device=None):
    return Tensor(np.empty(dim, dtype=dtype), dtype=dtype, device=device)