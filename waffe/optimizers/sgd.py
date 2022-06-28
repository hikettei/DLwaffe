
import waffe as wf
from .optimizer import OptimizerBase

class SGD(OptimizerBase):
	def update(self, param):
		grad = param.grad
		if grad is None:
			return
		param -= grad * self.lr