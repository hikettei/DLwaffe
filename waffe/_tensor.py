
from .str_format import wftensor_to_str
class Tensor():
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return wftensor_to_str(self.x)

    def to_list(self):
        return self.x
