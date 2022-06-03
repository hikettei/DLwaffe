
import waffe as wf

class TestTrainer(wf.Trainer):
    def __init__(self, model, dataset):
        pass

    @wf.Trainer.on_epoch
    def on_epoch(self, epoch_num):
        pass



model = MLP(720, 10)
dataset = TestDataset([1,2,3,4,5], [1,2,3,4,5], 2)

wrapper = TestTrainer(model, dataset)
wf.train(wrapper, epoch=60)
