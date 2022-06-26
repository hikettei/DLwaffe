import waffe as wf
import numpy as np
# utils

def mul2grad(tensor, mean=False):
    if tensor.data is None:
        total = len(tensor) if mean else 1
        return wf.Tensor(np.sum(tensor.detach())/total, device=tensor.device, is_constant=False).no_grad()
    else:
        return tensor.no_grad()


def collect_grads_for_deep_variables(target_variables, v_grads={}):
    # avoid target_variable itself
    if not target_variables in v_grads:
        for v in target_variables.variables:
            if v in v_grads.keys():
                v_grads[v].append(v.grad)
            else:
                v_grads[v] = [v.grad]

    if not target_variables.is_constant:
        for v in target_variables.variables:
            collect_grads_for_deep_variables(v, v_grads=v_grads)

def collect_grads_for_deep_variables0(target_variables, v_grads={}):
    for v in target_variables.variables:
        if v in v_grads.keys():
            if not v.grad in v_grads[v]:
                v_grads[v].append(v.grad)
        else:
            v_grads[v] = [v.grad]

    if not target_variables.is_constant:
        for v in target_variables.variables:
            collect_grads_for_deep_variables0(v, v_grads=v_grads)
# Backwards

def AddBackward(g, args, variables=[], tensor_self=None):
    # 1+1 or matrix + matrix
    v_grads = {}
    for i, exp in enumerate(args):
        exp.backward()
        collect_grads_for_deep_variables0(exp, v_grads=v_grads)

    for var, grads in v_grads.items():
        total = grads[0]
        for i in range(len(grads) - 1):
            if grads[0] == grads[1+i]: # when detect same grads... exa: test2
                break
            total = (total + grads[1+i]).no_grad()
        if total is not None:
            var.grad = total

def _AddBackward0(tensor):
    # matrix + 1
    def AddBackward0(g, args, variables=[], tensor_self=None):
        v_grads = {}
        for i, exp in enumerate(args):
            exp.backward()
            collect_grads_for_deep_variables(exp, v_grads=v_grads)

        for var, grads in v_grads.items():
            total = grads[0]
            for i in range(len(grads) - 1):
                if grads[0] == grads[1+i]: # when detect same grads... exa: test2
                    break
                total += (total + grads[1+i]).no_grad()
            if total is not None:
                var.grad = total.no_grad()#(total * len(tensor)).no_grad()
    return AddBackward0

def _SumBackward(tensor, mean=False):
    def _SumBackward_(g, args, variables=[], tensor_self=None):
        v_grads = {}

        tensor.backward()
        collect_grads_for_deep_variables(tensor, v_grads=v_grads)

        for var, grads in v_grads.items():
            total = grads[0]
            for i in range(len(grads) - 1):
                total += grads[1+i]
            if total is not None:
                if total.data is None:
                    var.grad = mul2grad(total, mean=mean)
                else:
                    if tensor.is_data():
                        var.grad = total if mean else total
                    else:
                        var.grad = total if mean else (total * len(tensor)).no_grad()
    return _SumBackward_

def _MeanBackward(tensor):
    def MeanBackward(*args, **kwargs):
        return _SumBackward(tensor, mean=True)(*args, **kwargs)
    return MeanBackward

def MulBackward(*args, variables=[], wrap_tensor=None, div_grad=False):
    # len(variables) must be 2
    x = variables[0]
    y = variables[1]

    t = args[0]
    if x.is_constant and y.is_constant:
        if x.is_data() and y.is_data():
            x.grad = t / x # == y
            y.grad = t / y # == x
        else:
            vec = x if y.is_data() else y
            con = y if y.is_data() else x
            con.grad = vec
    else:
        constant_variable_list = []
        v_grads = []
        grads = {}

        for v in [x, y]:
            collect_grads_for_deep_variables(v, v_grads=grads)

        constant_variable_list = list(grads.keys())

        x_grads = [0.] * len(constant_variable_list)
        y_grads = [0.] * len(constant_variable_list)

        x.backward()
        for i, v in enumerate(constant_variable_list):
            x_grads[i] = v.grad if v.grad is not None else wf.Tensor(0., device=x.device)

        # reset grads
        [v.reset_grad_value() for v in constant_variable_list]

        y.backward()
        for i, v in enumerate(constant_variable_list):
            y_grads[i] = v.grad if v.grad is not None else wf.Tensor(0., device=x.device)

        [v.reset_grad_value() for v in constant_variable_list]

        for i, v in enumerate(constant_variable_list):
            x_grad = x_grads[i]
            y_grad = y_grads[i]
            if div_grad:
                v_grads.append((x_grad * y - x * y_grad)/(y*y))
            else:
                v_grads.append(x * y_grad + x_grad * y)

        for v, v_grad in zip(constant_variable_list, v_grads):
            v.grad = v_grad.no_grad()

        return constant_variable_list

def DivBackward(*args, variables=[], wrap_tensor=None):
    return MulBackward(*args, variables=variables, wrap_tensor=wrap_tensor, div_grad=True)

def DerivBackward(*args, variables=[], tensor_self=None):
    g_x = args[0] # g(x) at f(g(x))
    d_f = args[1][0] # f'

    if g_x is not None:
        g_x.backward()
        v_grads = {}
        for v in variables:
            collect_grads_for_deep_variables(v, v_grads=v_grads)
        variables = v_grads.keys()
        for variable in variables:
            variable.grad = wf.Tensor(1., device=variable.device).no_grad() if variable.grad is None else variable.grad
            variable.grad = (d_f(g_x) * variable.grad).no_grad()
    else:
        d_f(tensor_self, variables=variables)

def _ConstantBackward(tensor_base):
    def ConstantBackward(*args, variables=[]):
        for v in args:
            if v == tensor_base:
                v.grad = wf.Tensor(1., extend=tensor_base).no_grad()
            else:
                v.grad = wf.Tensor(0., extend=tensor_base).no_grad()
    return ConstantBackward

def _PowBackward(tensor, k):
    def PowBackward(x):
        return wf.Tensor(tensor.detach() * k, device=tensor.device).no_grad()
    return PowBackward