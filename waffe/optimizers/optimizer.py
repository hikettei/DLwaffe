
class OptimizerBase():
	def __init__(self, model, lr=1e-4):
		self.model = model
		self.params = model.parameter_variables()
		self.lr = lr
	
	def step(self):
		for param in self.model.parameter_variables():#self.params:
			self.update(param[0], param[1])

	def update(self, status, set_val):
		pass

	def zero_grad(self):
		for param in self.params:
			param[2]()