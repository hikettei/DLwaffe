
import waffe as wf
from .optimizer import OptimizerBase

class SGD(OptimizerBase):
	def update(self, status, set_val):
		grad = status.grad
		if grad is None:
			return
		set_val(status - grad * self.lr)