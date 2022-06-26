import waffe as wf

import torch

import numpy as np

device = wf.get_device("device:1")

N = 10

a = np.random.rand(N)*30-15
b = float(np.random.randint(N))
c = float(np.random.randint(N))

a1 = wf.Tensor(a, device=device)
b1 = wf.Tensor(b, device=device)
c1 = wf.Tensor(c, device=device)

a2 = torch.tensor(a, requires_grad=True)
b2 = torch.tensor(b, requires_grad=True)
c2 = torch.tensor(c, requires_grad=True)

def reset():
	global a1, b1, c1, a2, b2, c2
	a1 = wf.Tensor(a, device=device)
	b1 = wf.Tensor(b, device=device)
	c1 = wf.Tensor(c, device=device)

	a2 = torch.tensor(a, requires_grad=True)
	b2 = torch.tensor(b, requires_grad=True)
	c2 = torch.tensor(c, requires_grad=True)

def assure(name="", dig=100, display_result=False):
	err = False
	for x in [[b1.grad, b2.grad], [c1.grad, c2.grad]]:
		if not err:
			if x[0] is None or x[1] is None:
				err = not x[0] is None and x[1] is None
			else:
				err = not int(x[0].detach() * dig) ==  int(x[1].detach() * dig)

	if not err:
		print("{}: No Error".format(name))
		if not display_result:
			reset()
			return
	print("==={}:===========".format(name))
	for x in [[b1.grad, b2.grad], [c1.grad, c2.grad]]:
		print("Waffe:{}".format(x[0]))
		print("Torch:{}".format(x[1]))

	reset()

def do_test(waffe_exp, torch_exp, name="Test", display_result=False):
	z1 = waffe_exp.mean()
	z2 = torch_exp.mean()
	z1.backward()
	z2.backward()
	assure(name=name, display_result=display_result)

do_test(wf.sin(a1 * b1),
		torch.sin(a2 * b2),
		name="Test1")

do_test(wf.sin(a1 * b1) * b1,
		torch.sin(a2 * b2) * b2,
		name="Test2")

do_test(wf.sin(a1 + a1),
		torch.sin(a2 + a2),
		name="Test3")

do_test(wf.sin(c1 + b1) + a1,
		torch.sin(c2 + b2) + a2,
		name="Test4")

do_test(wf.sin(a1) / wf.cos(b1),
		torch.sin(a2) / torch.cos(b2),
		name="Test5")

do_test(wf.cos(wf.sin(wf.sin(a1))) * wf.cos(b1),
		torch.cos(torch.sin(torch.sin(a2))) * torch.cos(b2),
		name="Test6")

do_test(a1**2 + b1**2,
		a2**2 + b2**2, name="Test7")

do_test(wf.sin(a1**2) + wf.cos(b1**2),
		torch.sin(a2**2) + torch.cos(b2**2), name="Test8")