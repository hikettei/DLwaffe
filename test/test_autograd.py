import waffe as wf

import torch

device = wf.get_device("device:1")

a = wf.Tensor(2, device=device)
b = wf.Tensor(2, device=device)
y = a + b

print(y)