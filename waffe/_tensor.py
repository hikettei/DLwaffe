
import waffe as wf
from .tensor import utils
from . import _backwards as bw

from .str_format import wftensor_to_str
import pyopencl as cl
import pyopencl.array as clarr

from .kernel import MAT_KERNELS

import numpy as np

DTYPES = {np.float16 : 'half', np.float32 : 'float', np.float64 : 'double'}

class WaffeDevice():
    def __init__(self, cl_device, **kwargs):
        self.DTYPE = kwargs['DTYPE'] if 'DTYPE' in kwargs else np.float32

        assert self.DTYPE in DTYPES

        floatX = DTYPES[self.DTYPE]
        self.TSM  = kwargs['TSM']  if 'TSM'  in kwargs else 1
        self.TSN  = kwargs['TSN']  if 'TSN'  in kwargs else 1
        self.TSK  = kwargs['TSK']  if 'TSK'  in kwargs else 1
        self.TS   = kwargs['TS']   if 'TS'   in kwargs else 1
        self.WPTM = kwargs['WPTM'] if 'WPTM' in kwargs else 1
        self.WPTN = kwargs['WPTN'] if 'WPTN' in kwargs else 1

        self.cl_device = cl_device
        self.ctx      = cl.Context([cl_device])
        self.queue    = cl.CommandQueue(self.ctx)

        options = "-DTSM={} -DTSN={} -DTSK={} -DWPTM={} -DWPTN={} -DfloatX={} -DTS={} -cl-mad-enable -cl-fast-relaxed-math".format(
            self.TSM, self.TSN, self.TSK, self.WPTM, self.WPTN, floatX, self.TS)

        self.prg  = cl.Program(self.ctx, MAT_KERNELS).build(options)
        self.name = cl_device.name


def is_data(x):
    return isinstance(x, (float, int)) or type(x) in DTYPES.keys()

def register_derivative(tensor, f, g, variables=None, deep_variables=True):
    #g... 次にBackwardするtensor
    #self.variables list of 変数
    if tensor.requires_grad:
        tensor.backwards = {"grad_fn":lambda self: bw.DerivBackward(g, [f], variables=self.variables, tensor_self=self, deep_variables=deep_variables),
                            "grad_name":f.__name__}

        if variables is not None:
            register_variables(tensor, variables)

def register_backwards_node(tensor, ident, *args, variables=None):
    if tensor.requires_grad:
        tensor.backwards = {"grad_fn":lambda self: ident(None, list(args), variables=self.variables, tensor_self=self),
                            "grad_name": ident.__name__}
        if variables is not None:
            register_variables(tensor, variables)

def register_variables(tensor, variables):
    if tensor.requires_grad:
        tensor.variables = variables

def create_res_buffer(tensor):
    M = np.int32(tensor.dim()[0])
    N = np.int32(tensor.dim()[1])

    cpu_earr = np.empty((M, N), dtype=tensor.device.DTYPE)
    resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
    res      = Tensor(cpu_earr, device=tensor.device, x_buf=resbuff, extend=tensor, is_constant=False)

    return (int(M), int(N)), (int(tensor.device.TS), int(tensor.device.TS)), M, N, res

def create_res_buffer_t(tensor):
    N = np.int32(tensor.dim()[0])
    M = np.int32(tensor.dim()[1])

    cpu_earr = np.empty((M, N), dtype=tensor.device.DTYPE)
    resbuff  = cl.Buffer(tensor.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
    res      = Tensor(cpu_earr, device=tensor.device, x_buf=resbuff, extend=tensor, is_constant=False)

    return (int(M), int(N)), (int(tensor.device.TS), int(tensor.device.TS)), M, N, res

class Tensor():
    #2回backwardとかしてないよね？
    def __init__(self, x, x_buf=None, extend=None, dtype=None, device=None, is_constant=True, requires_grad=True):
        """
        Exa: wf.Tensor([1,2,3])
        Arguments:
            x ... the type of x is as follows: list, numpy.array, numpy.ndarray

        is_constant=FalseであるTensorで微分できない。
        """

        assert isinstance(x, list) or is_data(x) or type(x).__module__ == np.__name__, "Not supported data type."        


        if device is None:
            if extend is None:
                #assert False
                print("Warning: missing device")
            device = wf.get_device("device:0") # In default, use cpu

        self.data = None

        if is_data(x): # 1x1の行列として扱う
            self.data = device.DTYPE(x)
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

        self.requires_grad = requires_grad
        self.device = device
        self.dtype  = dtype
        self.backwards = None

        register_derivative(self, bw._ConstantBackward(self), None)

        self.is_constant = is_constant
        self.variables = [self] # 
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
        return wftensor_to_str(self)

    def __matmul__(self, y):
        return wf.matmul(self, y)

    def __add__(self, y):
        self.sync()
        y.sync()

        if is_data(y):
            y_data = y
        else:
            y_data = y.data

        if self == y:
            return self.__mul__(Tensor(2., device=self.device))

        self_data_ex = self.data is not None
        y_data_ex = y_data is not None

        # [[1,2,3], [4,5,6], [7,8,9]] + [1,2,3]

        if self.dim() != y.dim() and not (self_data_ex or y_data_ex):
            if self.dim()[1] == y.dim()[1]:
                K = np.int32(y.dim()[1])
                gsize, lsize, M, N, res = create_res_buffer(self)
                event = self.device.prg.matpluscols(self.device.queue, gsize, lsize, M, N, K, self.x_buf, y.x_buf, res.x_buf)
                cl.wait_for_events([event, ])
                register_backwards_node(res, bw._AddBackward0(self), self, y, variables=[self, y])
                return res

            if self.dim()[0] == y.dim()[0]: #???
                self = self.transpose()
                K = np.int32(y.dim()[1])
                gsize, lsize, M, N, res = create_res_buffer(self)
                event = self.device.prg.matpluscols(self.device.queue, gsize, lsize, M, N, K, self.x_buf, y.x_buf, res.x_buf)
                cl.wait_for_events([event, ])
                register_backwards_node(res, bw._AddBackward0(self), self, y, variables=[self, y])
                return res

        if self_data_ex or y_data_ex:
            if self_data_ex and y_data_ex:
                x = self.data + y_data
                t = Tensor(x, dtype=self.dtype, device=self.device, extend=self, is_constant=False)
                register_backwards_node(t, bw.AddBackward, self, y, variables=[self, y])
                return t
            else:
                if self_data_ex:
                    vec = y
                    k = self.data
                else:
                    vec = self
                    k = y_data
                k  = self.device.DTYPE(k)
                gsize, lsize, M, N, res = create_res_buffer(vec)

                event = vec.device.prg.addk(vec.device.queue, gsize, lsize, M, N, k, vec.x_buf, res.x_buf)
                cl.wait_for_events([event, ])
                register_backwards_node(res, bw._AddBackward0(vec), self, y, variables=[self, y])
                self.sync()
                return res      
        else:
            assert self.dim()[0] == y.dim()[0], "{}, {}".format(self.dim(), y.dim())
            assert self.dim()[1] == y.dim()[1], "{}, {}".format(self.dim(), y.dim())
            gsize, lsize, M, N, res = create_res_buffer(self)
            event = self.device.prg.matsum(self.device.queue, gsize, lsize, M, N, self.x_buf, y.x_buf, res.x_buf)
            cl.wait_for_events([event, ])
            register_backwards_node(res, bw.AddBackward, self, y, variables=[self, y])
            self.sync()
            return res

    def __sub__(self, y):
        return self.__add__(Tensor(-1, device=self.device) * y)

    def __mul__(self, y, reciprocal=False):
        if self == y:
            # tmp
            return self.__pow__(2)

        self.sync()

        if is_data(y):
            y_data = y
        else:
            y_data = y.data

        self_data_ex = self.data is not None
        y_data_ex = y_data is not None

        if not self_data_ex and not y_data_ex: # matrix * matrix
            if reciprocal:
                res = Tensor(self.detach() / y.detach(), device=self.device, extend=self, is_constant=False)
            else:
                res = Tensor(self.detach() * y.detach(), device=self.device, extend=self, is_constant=False)
            backward = bw.DivBackward if reciprocal else bw.MulBackward
            register_derivative(res, backward, None, variables=[self, y])
            return res

        if self_data_ex or y_data_ex:
            if self_data_ex and y_data_ex:
                if reciprocal:
                    x = self.data / y_data
                else:
                    x = self.data * y_data
                t = Tensor(x, dtype=self.dtype, device=self.device, extend=self, is_constant=False)

                backward = bw.DivBackward if reciprocal else bw.MulBackward
                register_derivative(t, backward, None, variables=[self, y])
                return t
            else:
                if self_data_ex:
                    vec = y
                else:
                    vec = self
                k = self.device.DTYPE(self.data if self.data is not None else y_data)
                gsize, lsize, M, N, res = create_res_buffer(vec)
                if reciprocal:
                    event = vec.device.prg.matk(vec.device.queue, gsize, lsize, M, N, 1/k, vec.x_buf, res.x_buf)
                    cl.wait_for_events([event, ])
                else:
                    event = vec.device.prg.matk(vec.device.queue, gsize, lsize, M, N, k, vec.x_buf, res.x_buf)
                    cl.wait_for_events([event, ])
                backward = bw.DivBackward if reciprocal else bw.MulBackward
                register_derivative(res, backward, None, variables=[self, y])
                self.sync()
                return res

    def __truediv__(self, y):
        return self.__mul__(y, reciprocal=True)

    def __pow__(self, k):
        assert isinstance(k, int)
        res = Tensor(self.detach() ** k, device=self.device, is_constant=False)
        register_derivative(res, bw._PowBackward(self, k), self, variables=self.variables)
        return res

    def __len__(self):
        return len(self.detach())

    def sum(self):
        total = np.sum(self.detach())
        res = Tensor(total, device=self.device, is_constant=False)
        res.sync()

        register_backwards_node(res, bw._SumBackward(self), self, variables=self.variables)
        return res

    def mean(self):
        s = self.sum()
        t = Tensor(len(self), device=self.device, is_constant=False)
        res = s / t
        register_backwards_node(res, bw._MeanBackward(self), self, variables=self.variables)
        return res

    def transpose(self):
        M = np.int32(self.dim()[0])
        N = np.int32(self.dim()[1])
        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Tensor(cpu_earr, device=self.device, x_buf=resbuff, extend=self, is_constant=False)

        gsize = (int(N), int(M))
        lsize = (int(self.device.TS), int(self.device.TS))

        event = self.device.prg.transpose(self.device.queue, gsize, lsize, N, M, self.x_buf, res.x_buf)
        cl.wait_for_events([event, ])
        register_derivative(res, bw._TransposeBackward(self), self, variables=self.variables)
        return res

    def expand_dims(self, dim):
        return wf.expand_dims(self, dim)

    def reshape(self, dim):
        return wf.reshape(self, dim)

    def backward(self):
        self.sync()
        #assert self.data is not None, "grad can be implicitly created only for scalar outputs"
        self.backwards["grad_fn"](self)
        return None

    def detach(self):
        x = np.empty(self.shape, dtype=self.device.DTYPE)
        event = cl.enqueue_copy(self.device.queue, x, self.x_buf)
        cl.wait_for_events([event, ])
        if self.d_shape == 1:
            return np.reshape(x[:self.shape[0], :self.shape[1]], -1).astype(self.dtype)[0]
        else:
            try:
                return np.reshape(x[:self.shape[0], :self.shape[1]], self.d_shape).astype(self.dtype)
            except ValueError: #???
                return np.reshape(x[:self.shape[0], :self.shape[1]], self.shape).astype(self.dtype)
                return x.reshape(-1)

    def write_mem(self, x_tensor):
        event = cl.enqueue_write_buffer(
                queue=self.device.queue,
                mem=x_tensor.x_buf,
                hostbuf=self.x_buf)

        cl.wait_for_events([event, ])
        self.sync()
        return self

    def dim(self):
        return self.shape

    def sync(self):
        x = self.detach()
        if is_data(x):
            self.data = x
        return self

    def no_grad(self):
        self.requires_grad = False
        return self

    def zero_grad(self):
        # 勾配をリセット
        self = Tensor(self.detach(), extend=self)
        return self

    def reset_grad_value(self):
        self.grad = None
        return self

    def is_data(self):
        return is_data(self.detach())

class no_grad():
    pass

def empty(dim, dtype=None, device=None):
    return Tensor(np.empty(dim, dtype=dtype), dtype=dtype, device=device)

def randn(*args, device=None):
    return Tensor(np.random.randn(*args), device=device)

def randint(low, high, dim, device=None):
    return Tensor(np.random.randint(low, high, dim), device=device)