
import waffe as wf
import numpy as np


device = wf.get_device("device:1")

x1 = np.array([[1,2,3], [4,5,6]])
x2 = [[1,2,3], [4,5,6], [7,7,9]]

xt1 = wf.Tensor(x1)
xt2 = wf.Tensor(x2)

print(xt1)
print(xt2)
