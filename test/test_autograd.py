import waffe as wf

import torch

device = wf.get_device("device:1")

a, b, c = 1.0, 9.0, 3.0

a1 = wf.Tensor(a, device=device)
b1 = wf.Tensor(b, device=device)
c1 = wf.Tensor(c, device=device)

a2 = torch.tensor(a, requires_grad=True)
b2 = torch.tensor(b, requires_grad=True)
c2 = torch.tensor(c, requires_grad=True)

def t1(name="Test1"):
	z1 = wf.sin(a1 * b1) * wf.cos(a1 * b1)
	z2 = torch.sin(a2 * b2) * torch.cos(a2 * b2)
	z1.backward()
	z2.backward()
	print("==={}:======".format(name))
	for x in [[a1.grad, a2.grad], [b1.grad, b2.grad], [c1.grad, c2.grad]]:
		print("Waffe:{}".format(x[0]))
		print("Torch:{}".format(x[1]))


def t2(name="Test2"):
	z1 = wf.sin(a1 * b1) * wf.cos(a1 * b1) + c1
	z2 = torch.sin(a2 * b2) * torch.cos(a2 * b2) + c2
	z1.backward()
	z2.backward()
	print("==={}:======".format(name))
	for x in [[a1.grad, a2.grad], [b1.grad, b2.grad], [c1.grad, c2.grad]]:
		print("Waffe:{}".format(x[0]))
		print("Torch:{}".format(x[1]))


def t3(name="Test3"):
	z1 = wf.sin(a1 + a1) + a1
	z2 = torch.sin(a2 + a2) + a2
	z1.backward()
	z2.backward()
	print("==={}:======".format(name))
	for x in [[a1.grad, a2.grad], [b1.grad, b2.grad], [c1.grad, c2.grad]]:
		print("Waffe:{}".format(x[0]))
		print("Torch:{}".format(x[1]))


def t4(name="Test4"):
	z1 = wf.sin(a1 + b1) + a1
	z2 = torch.sin(a2 + b2) + a2
	z1.backward()
	z2.backward()
	print("==={}:======".format(name))
	for x in [[a1.grad, a2.grad], [b1.grad, b2.grad], [c1.grad, c2.grad]]:
		print("Waffe:{}".format(x[0]))
		print("Torch:{}".format(x[1]))

def reset():
	global a1, b1, c1, a2, b2, c2
	a1 = wf.Tensor(a, device=device)
	b1 = wf.Tensor(b, device=device)
	c1 = wf.Tensor(c, device=device)

	a2 = torch.tensor(a, requires_grad=True)
	b2 = torch.tensor(b, requires_grad=True)
	c2 = torch.tensor(c, requires_grad=True)


t3()
reset()
t4()