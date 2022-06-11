import waffe as wf

import torch

device = wf.get_device("device:1")

a = wf.Tensor(2.0, device=device)
b = wf.Tensor(9.0, device=device)
c = wf.Tensor(3.0, device=device)

#z = wf.log(a * b) * wf.sin(a * b)

z = wf.sin(a * b) * wf.cos(a * b)
# {a * wf.cos(a * b)} * 1 + 0 * wf.sin(a * b)
# 
z.backward()
print(a.grad)
print(b.grad)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(9.0, requires_grad=True)
c = torch.tensor(3.0, requires_grad=True)
z = torch.sin(a * b) * torch.cos(a * b)
z.backward()

print(a.grad)
print(b.grad)