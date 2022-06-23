import waffe as wf
import waffe.nn.functional as F

class MLP(wf.Model):
    def __init__(self, embedding_dim, hidden_dim, device=None):
        self.layer1 = wf.nn.Linear(embedding_dim, hidden_dim, device=device)

    @wf.Model.on_batch
    def on_batch(self, x):
        x = self.layer1(x)
        return wf.sigmoid(x)


device = wf.get_device("device:1")

model = MLP(3, 3, device=device)
x = wf.randn(3, 3, device=device)
out = model(x)
out.backward()
print(x.grad)