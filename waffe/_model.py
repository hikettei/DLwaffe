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
        def _zero_grad(var_name, status):
            def __zero_grad():
                setattr(getattr(self, var_name), "grad", None)
                #setattr(self, var_name, status.zero_grad(is_param=True))
            return __zero_grad
        def _refresh_param(status):
            def refresh_param(x):
                status.set_tensor_data(x)
            return refresh_param

        for var_name, status in self.__dict__.items():
            if isinstance(status, wf.Model):
                parameters += status.parameter_variables()
            if isinstance(status, wf.Tensor):
                if status.is_param:
                    parameters.append([status, _refresh_param(status), _zero_grad(var_name, status)])
        return parameters