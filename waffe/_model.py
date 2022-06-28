import waffe as wf
from ._events import ModuleEventListener

class Model(ModuleEventListener):
    def __call__(self, *args, **kwargs):
        result = self.on_batch(*args, **kwargs)
        
        if type(result) == 'func':
            return None # a on_batch haven't implemented yet
        else:
            return result

    def parameters(self):
    	parameters = {self.__class__: []}
    	for var_name, status in self.__dict__.items():
    		if isinstance(status, wf.Model):
    			parameters[self.__class__].append(status.parameters())

    		if isinstance(status, wf.Tensor):
    			if status.is_param:
    				parameters[self.__class__].append(status)

    	return parameters

    def parameter_variables(self):
    	parameters = []
    	for var_name, status in self.__dict__.items():
    		if isinstance(status, wf.Model):
    			parameters += status.parameter_variables()
    		if isinstance(status, wf.Tensor):
    			if status.is_param:
    				parameters.append(status)
    	return parameters