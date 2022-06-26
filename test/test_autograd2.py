import waffe as wf
import waffe.nn.functional as F

import numpy as np
import torch


device = wf.get_device("device:1")

N = 10
x1 = np.random.rand(N)*30-15
y1 = 2*x1 + np.random.randn(N)*5

x = wf.Tensor(x1, device=device)
y = wf.Tensor(y1, device=device)

w = wf.Tensor(1.0, device=device)
b = wf.Tensor(0.0, device=device)

l = wf.sin((x * w + b)-y)
z = l.mean()
print("++++")
print(z)
z.backward()
print(b.grad)
print(w.grad)

x = torch.tensor(x1, requires_grad=True)
y = torch.tensor(y1, requires_grad=True)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
l = torch.sin((x * w + b)-y)
z = l.mean()
print("++++")
print(z)
z.backward()
print(b.grad)
print(w.grad)


a = wf.Tensor(1.0, device=device)
b = wf.Tensor(2.0, device=device)
c = wf.Tensor(9.0, device=device)

l = ((a * b + c) - a)**2
l.backward()
print(a.grad)
print(b.grad)
print(c.grad)

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = torch.tensor(9.0, requires_grad=True)

l = ((a * b + c) - a)**2
l.backward()
print(a.grad)
print(b.grad)
print(c.grad)