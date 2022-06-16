import os

CL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'kernel')

file_names = ["math.c", "dfunctions.c", "functions.c"]
MAT_KERNELS = ""

for file_name in file_names:
	with open(os.path.join(CL_DIR, file_name), 'r') as _f:
		MAT_KERNELS += _f.read() + "\n"