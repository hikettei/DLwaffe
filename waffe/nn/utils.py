
import waffe as wf
from . import functional as F 

class Linear(wf.Model):
    def __init__(self, in_features, out_features, bias=True, device=None):
        factory_kwargs = {'device': device}
        self.in_features = in_features
        self.out_features = out_features
        self.weights = wf.empty((in_features, out_features), device=device)
        self.bias    = wf.empty(out_features, **factory_kwargs)

    @wf.Model.on_batch
    def on_batch(self, x):
        return F.linear(x, self.weights, self.bias)