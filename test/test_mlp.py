import waffe as wf
import waffe.nn.functional as F
import waffe.optimizers as opts

from pprint import pprint
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class TestDataset(wf.Dataset):
    def __init__(self, x, y, batch_size, device=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.i = 0
        self.device = device

    @wf.Dataset.on_batch
    def on_batch(self):
        f, e = self.i, self.i + self.batch_size
        x, y = self.x[f:e].reshape(-1, 28*28), self.y[f:e]
        self.i += self.batch_size
        y_ = []
        for l in y:
            tmp = [0.] * 10
            tmp[l] = 1.
            y_.append(tmp)
        return wf.Tensor(x, device=device), wf.Tensor(y_, device=device)

    def is_next(self):
        return self.i < len(self.x)

    def reset(self):
        self.i = 0

    def total(self):
        return len(self.x)


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

class MLPTrainer(wf.Trainer):
    def __init__(self, model, criterion, optimizer, lr=1e-4):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def on_batch(self, data):
        x, y = data
        out = self.model(x)
        loss = self.criterion(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

# prepare mnist
mnist_train = datasets.mnist.MNIST("./data", download=True, transform=transforms.ToTensor(), train=True)
mnist_val   = datasets.mnist.MNIST("./data", download=True, transform=transforms.ToTensor(), train=False)

train_img    = mnist_train.data.numpy()[:100]
train_labels = mnist_train.targets.numpy()[:100]
valid_img    = mnist_val.data.numpy()
valid_labels = mnist_val.targets.numpy()

device = wf.get_device("device:1")

model = MLP(device=device)
dataset = TestDataset(train_img,
                      train_labels,
                      3, device=device)

trainer = MLPTrainer(model, wf.nn.cross_entropy, opts.SGD(model), lr=1e-3)
wf.train(trainer, dataset, epoch_num=1, save_model_each=1)

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