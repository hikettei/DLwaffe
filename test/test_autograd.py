import waffe as wf

import torch

device = wf.get_device("device:1")

a = wf.Tensor(1.0, device=device)
b = wf.Tensor(2.0, device=device)
#c = wf.Tensor(3, device=device)

#z = wf.log(a * b) * wf.sin(a * b)
z = wf.sin(a * b)
z.backward()

print(a.grad)
print(b.grad)

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
z = torch.sin(a * b)
z.backward()

print(a.grad)
print(b.grad)