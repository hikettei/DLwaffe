
import waffe as wf
from . import functional as F
import numpy as np

class Linear(wf.Model):
    def __init__(self, in_features, out_features, bias=True, device=None):
        factory_kwargs = {'device': device}
        self.in_features = in_features
        self.out_features = out_features
        self.weights = wf.Tensor(0.01 * np.random.randn(in_features, out_features), device=device).as_param()
        self.bias    = wf.Tensor(np.zeros(out_features), device=device).as_param() if bias else None
        #self.weights = wf.randn(in_features, out_features, device=device).as_param()
        #self.bias    = wf.randn(out_features, **factory_kwargs).as_param() if bias else None

    @wf.Model.on_batch
    def on_batch(self, x):
        return F.linear(x, self.weights, self.bias)

class Dense(wf.Model):
    def __init__(self, in_features, out_features, bias=True, activation=wf.relu, device=None):
        factory_kwargs = {'device': device, 'bias':bias}
        self.linear = Linear(in_features, out_features, **factory_kwargs)
        self.activation = activation

    @wf.Model.on_batch
    def on_batch(self, x):
        return self.activation(self.linear(x))