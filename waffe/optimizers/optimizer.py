
class OptimizerBase():
	def __init__(self, model, lr=1e-4):
		self.model = model
		self.lr = lr
	
	def step(self):
		self.params = self.model.parameter_variables()
		for param in self.params:
			self.update(param[0], param[1])

	def update(self, status, set_val):
		pass

	def zero_grad(self):
		self.params = self.model.parameter_variables()
		for param in self.params:
			param[2]()