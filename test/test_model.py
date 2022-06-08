import waffe as wf

class MLP(wf.Model):
    def __init__(self, embedding_dim, hidden_dim, device=None):
        self.layer1 = wf.nn.Linear(embedding_dim, hidden_dim, device=device)

    @wf.Model.on_batch
    def on_batch(self, x):
        return self.layer1(x)

device = wf.get_device("device:1")

model = MLP(720, 10, device=device)
x = wf.randn(720, 1, device=device)

out = model(x)
print(out)