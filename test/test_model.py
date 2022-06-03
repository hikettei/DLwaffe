
import waffe as wf

class MLP(wf.Model):
    def __init__(self, embedding_dim, hidden_dim):
        self.layer1 = wf.nn.Linear(embedding_dim, hidden_dim)

    @wf.Model.on_batch
    def on_batch(self, x):
        return self.layer1(x)


model = MLP(720, 10)
x = wf.empty((720, 1))
out = model(x)
print(out)
