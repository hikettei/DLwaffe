
import waffe as wf

class TestDataset(wf.Dataset):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.i = 0

    @wf.Dataset.on_batch
    def on_batch(self):
        f, e = self.i, self.i + self.batch_size
        x, y = self.x[f:e], self.y[f:e]
        self.i += self.batch_size
        return x, y

dataset = TestDataset(list(range(0, 100)), list(range(0, 100)), 10)

for i in range(10):
    print(dataset.on_batch())
