
import waffe as wf
from . import functional as F 

class Linear(wf.Model):
    def __init__(self, in_features, out_features, bias=True, device=None):
        factory_kwargs = {'device': device}
        self.in_features = in_features
        self.out_features = out_features
        self.weights = wf.empty((in_features, out_features), device=device)
        self.bias    = wf.empty(out_features, **factory_kwargs) if bias else None

    @wf.Model.on_batch
    def on_batch(self, x):
        return F.linear(x, self.weights, self.bias)

class Dense(wf.Model):
    def __init__(self, in_features, out_features, bias=True, activation=wf.sigmoid, device=None):
        factory_kwargs = {'device': device, 'bias':bias}
        self.linear = Linear(in_features, out_features, **factory_kwargs)
        self.activation = activation

    @wf.Model.on_batch
    def on_batch(self, x):
        return self.activation(self.linear(x))