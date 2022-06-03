
import waffe as wf

class Linear(wf.Model):
    def __init__(self, in_featurees, out_features, bias=True):
        pass

    @wf.Model.on_batch
    def on_batch(self, x):
        return x
