
class OptimizerBase():
	def __init__(self, model, lr=1e-4):
		self.params = model.parameter_variables()
		self.lr = lr
	
	def step(self):
		for param in self.params:
			self.update(param)

	def update(self, param):
		pass

	def zero_grad(self):
		for param in self.params:
			param.zero_grad()