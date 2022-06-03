
import waffe as wf

from .str_format import wftensor_to_str
import pyopencl as cl
import pyopencl.array as clarr

import numpy as np

from enum import Enum

class Tensor():
    def __init__(self, x, dtype=None, device=None):
        """
        Exa: wf.Tensor([1,2,3])
        Arguments:
            x ... the type of x is as follows: list, numpy.array, numpy.ndarray
        """

        assert isinstance(x, list) or type(x).__module__ == np.__name__, "The first argument of wf.Tensor must be list or numpy list"        
        
        if isinstance(x, list):
            x = np.asarray(x)

        if dtype is None:
            dtype = x.dtype

        if device is None:
            device = wf.get_device("device:0") # Use CPU

        #Unpack WaffeDevice
        device = device.device

        device = cl.Context([device])
        
        queue = cl.CommandQueue(device)
        self.x_mem = clarr.Array(queue, x.shape, dtype=dtype)
        self.x_mem.set(x)

    def __str__(self):
        return wftensor_to_str(self.x_mem.get())

    def to_list(self):
        return self.x_mem.get()
