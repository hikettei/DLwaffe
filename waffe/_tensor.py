
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
        self.name = cl_device.name


def is_data(x):
    return isinstance(x, (float, int)) or type(x) in DTYPES.keys()

def register_derivative(tensor, f, g):
    #g... 次にBackwardする
    #self.variables list of 変数
    tensor.backwards = {"type":"d", "func":_deriv, "args":[f], "g":g}

def register_backwards_node(tensor, ident, *args):
    tensor.backwards = ({"type":"node", "func":ident, "args":list(args)})

def register_variables(tensor, variables):
    tensor.variables = variables

def view_node(tensor, nodes=[]):
    pass

def _add(*args, variables=[]):
    for arg in args[0]:
        arg.backward(variables)

def _mul(*args, variables=[]):
    for variable in variables:
        variable.grad = args[0] / variable

def _deriv(*args, variables=[], tensor_self=None):
    g_x = args[0] # g(x)
    d_f = args[1][0] # d_f

    if g_x is not None:
        g_x.variables = variables
        g_x.backward()

    for variable in variables:
        if g_x is None:
            d_f(tensor_self, variables=[variable]) # _mul
            variable.grad = variable.grad
        else:
            variable.grad = variable.grad * d_f(variable)
        #print(variable.grad)
        #print("grad")
        #print(variable)
        #print(variable.grad * d_f(variable))

class Tensor():
    def __init__(self, x, dtype=None, device=None, x_buf=None, extend=None):
        """
        Exa: wf.Tensor([1,2,3])
        Arguments:
            x ... the type of x is as follows: list, numpy.array, numpy.ndarray
        """

        assert isinstance(x, list) or is_data(x) or type(x).__module__ == np.__name__, "Not supported data type."        


        if device is None:
            device = wf.get_device("device:0") # In default, use cpu

        self.data = False

        if is_data(x): # 1x1の行列として扱う
            self.data = x
            x = np.asarray([[x]]).astype(device.DTYPE)
            self.d_shape = 1
        else:
            x = np.asarray(x).astype(device.DTYPE)
            self.d_shape = x.shape 

        if len(x.shape) <= 1:
            x = np.asarray([x]).astype(device.DTYPE)

        self.shape = x.shape

        if dtype is None:
            dtype = x.dtype

        # Restore device, dtype
        self.device = device
        self.dtype  = dtype
        self.backwards = None
        self.variables = []
        self.grad = None
        self.is_input = True

        if extend is not None:
            self.device = extend.device
            self.dtype  = extend.dtype
            self.d_shape = extend.d_shape
            self.is_input = False

        if x_buf is None:
            self.x_buf = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        else:
            self.x_buf = x_buf

    def __str__(self):
        return wftensor_to_str(self.detach())

    def __matmul__(self, y):
        assert self.dim()[0] == y.dim()[0]
        assert self.device.name == y.device.name

        M = np.int32(self.dim()[1])
        K = np.int32(self.dim()[0])
        N = np.int32(y.dim()[1])

        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff  = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res      = Tensor(cpu_earr, device=self.device, x_buf=resbuff, extend=self)

        #global_sizes = (int(M/self.device.WPTM), int(N/self.device.WPTN))
        #local_sizes = (int(self.device.TSM/self.device.WPTM), int(self.device.TSN/self.device.WPTN))
        #print(global_sizes)
        #print(local_sizes)
        #int(M/self.device.WPTM)

        heads = int(M/self.device.WPTM)
        if heads < 1:
            heads = 1
        heads = 1 # tmp
        event = self.device.prg.matmul(self.device.queue, (heads, ), None, M, N, K, self.x_buf, y.x_buf, res.x_buf)
        cl.wait_for_events([event, ])
        
        return res

    def __add__(self, y):
        assert self.dim()[0] == y.dim()[0]
        assert self.dim()[1] == y.dim()[1]
        M = np.int32(self.dim()[0])
        N = np.int32(self.dim()[1])

        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff  = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res      = Tensor(cpu_earr, device=self.device, x_buf=resbuff, extend=self)
        event = self.device.prg.matsum(self.device.queue, (1,), None, M, N, self.x_buf, y.x_buf, res.x_buf)
        cl.wait_for_events([event, ])
        register_backwards_node(res, _add, self, y)
        self.sync()
        return res

    def __sub__(self, y):
        assert y.dim()[0] == self.dim()[0]
        assert y.dim()[1] == self.dim()[1]
        M = np.int32(y.dim()[0])
        N = np.int32(y.dim()[1])

        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff  = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res      = Tensor(cpu_earr, device=self.device, x_buf=resbuff, extend=self)

        event = self.device.prg.matsubstract(self.device.queue, (1,), None, M, N, y.x_buf, self.x_buf, res.x_buf)
        cl.wait_for_events([event, ])
        self.sync()
        return res

    def __mul__(self, y):
        if self.data or y.data:
            if self.data and y.data:
                x = self.data * y.data
                t = Tensor(x, dtype=self.dtype, device=self.device, extend=self)
                register_derivative(t, _mul, None)
                register_variables(t, [self, y])
                return t
            else:
                if self.data:
                    vec = y
                else:
                    vec = self
                k  = np.int32(self.data or y.data)
                M = np.int32(vec.dim()[0])
                N = np.int32(vec.dim()[1])
                cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
                resbuff  = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
                res      = Tensor(cpu_earr, device=self.device, x_buf=resbuff, extend=self)

                event = vec.device.prg.matk(vec.device.queue, (1,), None, M, N, k, vec.x_buf, res.x_buf)
                cl.wait_for_events([event, ])
                self.sync()
                return res
        else:
            self.sync()
            return self.__matmul__(y)

    def __truediv__(self, y):
        return Tensor(self.detach() / y.detach(), extend=self)

    def __mod__(self, y):
        assert self.dim()[0] == y.dim()[0], "The mismatch shape"
        assert self.device.name == y.device.name, "The device doesn't correspond"
        M = np.int32(self.dim()[0])
        N = np.int32(self.dim()[1])
        K = np.int32(y.dim()[1] if len(y.dim()) > 1 else 1)
        gsize = (int(M), )
        event = self.device.prg.matpluscols(self.device.queue, gsize, None, M, N, K, self.x_buf, y.x_buf, self.x_buf)
        cl.wait_for_events([event, ])
        self.sync()
        return self

    def backward(self):
        b = self.backwards
        print(b)
        b["func"](b["g"], b["args"], variables=self.variables, tensor_self=self)
        return None

    def to_list(self):
        # x.to_list() returns list
        return self.detach()

    def detach(self):
        x = np.empty(self.shape, dtype=self.device.DTYPE)
        event = cl.enqueue_copy(self.device.queue, x, self.x_buf)
        cl.wait_for_events([event, ])
        if self.d_shape == 1:
            return np.reshape(x[:self.shape[0], :self.shape[1]], -1).astype(self.dtype)[0]
        else:
            return np.reshape(x[:self.shape[0], :self.shape[1]], self.d_shape).astype(self.dtype)

    def write_mem(self, x_tensor):
        event = cl.enqueue_write_buffer(
                queue=self.device.queue,
                mem=x_tensor.x_buf,
                hostbuf=self.x_buf)

        cl.wait_for_events([event, ])
        return self

    def dim(self):
        return self.shape

    def sync(self):
        x = self.detach()
        if is_data(x):
            self.data = x
        return self

def empty(dim, dtype=None, device=None):
    return Tensor(np.empty(dim, dtype=dtype), dtype=dtype, device=device)

def randn(*args, device=None):
    return Tensor(np.random.randn(*args), device=device)

def randint(low, high, dim, device=None):
    return Tensor(np.random.randint(low, high, dim), device=device)