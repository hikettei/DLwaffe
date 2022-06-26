import waffe as wf
import waffe.nn.functional as F

class MLP(wf.Model):
    def __init__(self, device=None):
        self.layer1 = wf.nn.Dense(28 * 28, 128, device=device)
        self.layer2 = wf.nn.Dense(128, 256, device=device)
        self.layer3 = wf.nn.Dense(256, 10, device=device)

    @wf.Model.on_batch
    def on_batch(self, x):
    	x = self.layer1(x)
    	x = self.layer2(x)
    	x = self.layer3(x)
    	return x


device = wf.get_device("device:1")

model = MLP(device=device)
x = wf.randn(3, 28*28, device=device)
out = model(x)
print(out.dim())

#out.backward()
#print(x.grad)
"""
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.ReLU(),
                                 nn.Linear(256, 10), nn.LogSoftmax(-1))
        
    def forward(self, x):
        return self.net(x)

model = MLP1()

train_loader = DataLoader(datasets.mnist.MNIST("./data", download=True, transform=transforms.ToTensor(), train=True), batch_size=32, shuffle=True)
val_loader = DataLoader(datasets.mnist.MNIST("./data", download=True, transform=transforms.ToTensor(), train=False), batch_size=32, shuffle=False)

for x, y in train_loader:
	a= model(x.reshape(-1, 28 * 28))
	print(a.shape)
	break
	"""