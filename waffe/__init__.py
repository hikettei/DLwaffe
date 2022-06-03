
import pyopencl as cl
import numpy as np

class WaffeDevice():
    def __init__(self, device):
        self.device = device


def render_all_devices_info():
    for platform in cl.get_platforms():
        for i, device in enumerate(platform.get_devices()):
            print(f"===[Device:{i}]================================")
            print("Default Device?: ", i == 0)
            print("Platform name:   ", platform.name)
            print("Platform version:", platform.version)
            print("Device name:     ", device.name)
            print("Device type:     ", cl.device_type.to_string(device.type))
            print("Device memory:   ", device.global_mem_size//1024//1024, 'MB')
            print("Device units:    ", device.max_compute_units)


def backends():
    device_list = {}
    for platform in cl.get_platforms():
        for i, device in enumerate(platform.get_devices()):
            device_list[f"device:{i}"] = WaffeDevice(device)
    return device_list

def get_device(name):
    device_list = backends()
    if name in device_list.keys():
        return device_list[name]

from ._model import Model
from ._dataset import Dataset
from ._trainer import Trainer
from ._tensor  import Tensor


# Module nn

from . import nn
