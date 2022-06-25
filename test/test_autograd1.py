import waffe as wf
import waffe.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch

N = 20
x_ = np.random.rand(N)*30-15
y_ = 2*x_ + np.random.randn(N)*5
 
x_ = x_.astype(np.float32)
y_ = y_.astype(np.float32)

x = torch.from_numpy(x_)
y = torch.from_numpy(y_)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 1.0e-4
 
def model(x):
	return w*x + b

def mse(p, y):
	return ((p-y)**2).mean()

losses = []
for epoch in tqdm(range(300)):
    p = model(x)
    
    loss = mse(p, y)
    loss.backward()
    
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr
        w.grad.zero_()
        b.grad.zero_()
 
    losses.append(loss.item())
    
print('loss = ', loss.item())
print('w    = ', w.item())
print('b    = ', b.item())



device = wf.get_device("device:1")

x = wf.Tensor(x_, device=device)
y = wf.Tensor(y_, device=device)

w = wf.Tensor(1.0, device=device)
b = wf.Tensor(0.0, device=device)

def model(x):
	return w*x + b

def mse(p, y):
	return ((p-y)**2).mean()

lr = 1.0e-4
 
losses = []
for epoch in tqdm(range(300)):
    p = model(x)
    
    loss = mse(p, y)
    loss.backward()
    
    #with torch.no_grad():
    w -= (w.grad * lr).no_grad()
    b -= (b.grad * lr).no_grad()
    w = w.zero_grad()
    b = b.zero_grad()
 
    # グラフ描画用
    losses.append(loss.detach())
    
print('loss = ', loss.detach())
print('w    = ', w.detach())
print('b    = ', b.detach())