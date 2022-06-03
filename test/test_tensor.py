
import waffe as wf
import numpy as np

from time import time

wf.render_all_devices_info()
device = wf.get_device("device:1")

x1 = np.array([[1,2,3], [4,5,6]])
x2 = [[1,2], [4,5], [6, 7]]

xt1 = wf.Tensor(x1)
xt2 = wf.Tensor(x2)

print(xt1.reshape((3, 2)))
print(xt2)

xt1.transpose()

print(xt1)

xt3 = wf.empty((2, 10))
print(xt3)

gpu_device = wf.get_device("device:1")

gpu_x = wf.empty((3000, 3000), device=gpu_device)
gpu_y = wf.empty((3000, 3000), device=gpu_device)

time1 = time()

for _ in range(10):
	gpu_x @ gpu_y

time2 = time()


print("Execution time of test with device:1 ", time2 - time1, "s")

cpu_device = wf.get_device("device:0")

cpu_x = wf.empty((3000, 3000), device=cpu_device)
cpu_y = wf.empty((3000, 3000), device=cpu_device)

time1 = time()
for _ in range(10):
	gpu_x @ gpu_y
time2 = time()

print("Execution time of test with device:0 ", time2 - time1, "s")

x = np.empty((3000, 3000))
y = np.empty((3000, 3000))

time1 = time()
for _ in range(10):
	x @ y
time2 = time()

print("Execution time of test with numpy ", time2 - time1, "s")