import waffe as wf
import waffe.nn.functional as F
from pprint import pprint

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
out.backward()
pprint(model.parameters())
