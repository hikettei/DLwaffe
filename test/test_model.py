
import waffe as wf

class MLP(wf.Model):
    def __init__(self, embedding_dim, hidden_dim):
        self.layer1 = wf.nn.Linear(embedding_dim, hidden_dim)

    @wf.Model.on_batch
    def on_batch(self, x):
        return self.layer1(x)


model = MLP(720, 10)
x = wf.Tensor([1,2,3,4,5,6,7,8,9])
out = model.on_batch(x)
print(out)
