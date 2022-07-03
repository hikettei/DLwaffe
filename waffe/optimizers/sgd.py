
import waffe as wf
from .optimizer import OptimizerBase

class SGD(OptimizerBase):
	def update(self, status, set_val):
		grad = status.grad
		if grad is None:
			return
		status.set_tensor_data(status - grad * self.lr)
		#set_val(status - grad * self.lr)