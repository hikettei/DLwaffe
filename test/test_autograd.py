import waffe as wf

import torch

device = wf.get_device("device:1")

a = wf.Tensor(2.0, device=device)
b = wf.Tensor(9.0, device=device)
c = wf.Tensor(3.0, device=device)

z = wf.sin(wf.sin(a * b)) * wf.cos(a * b) + c
z.backward()

print("==Test1:========")
print(a.grad)
print(b.grad)
print(c.grad)

z = wf.sin(wf.sin(a * b) + c)
z.backward()
print("==Test2:========")
print(a.grad)
print(b.grad)
print(c.grad)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(9.0, requires_grad=True)
c = torch.tensor(3.0, requires_grad=True)
z = torch.sin(torch.sin(a * b)) * torch.cos(a * b) + c
z.backward()
print("==Test1:========")
print(a.grad)
print(b.grad)
print(c.grad)

z = torch.sin(torch.sin(a * b) + c)
z.backward()
print("==Test2:========")
print(a.grad)
print(b.grad)
print(c.grad)