
from ._events import ModuleEventListener

class Model(ModuleEventListener):
    def __call__(self, *args, **kwargs):
        result = self.on_batch(*args, **kwargs)
        
        if type(result) == 'func':
            return None # a on_batch haven't implemented yet
        else:
            return result
