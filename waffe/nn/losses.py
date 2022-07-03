import waffe as wf

def _CrossEntropyBackward(x, t, tensors):
	def CrossEntropyBackward(args, variables=[]):
		pass
	return CrossEntropyBackward

def cross_entropy(x, t, delta=1e-7):
	"""
	Input:
		x: the tensor of possibility
		t: the answer labels
	"""
	coeff = wf.Tensor(-1.0 / max(len(x), 1), device=x.device)
	res = (t * wf.log(x + wf.Tensor(delta, device=x.device))).sum(keepdims=True) * coeff
	#wf.register_derivative(res_, _CrossEntropyBackward(x, t, res), None, variables=None)
	return res

def mse(p, y):
	return ((p-y)**2).mean()