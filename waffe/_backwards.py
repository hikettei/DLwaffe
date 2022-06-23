import waffe as wf
import numpy as np
# utils

def mul2grad(tensor):
    if tensor.data is None:
        return wf.Tensor(np.sum(tensor.detach()), device=tensor.device, is_constant=False).no_grad()
    else:
        return tensor.no_grad()

# Backwards

def AddBackward(g, args, variables=[], tensor_self=None):
    v_grads = {}
    for i, exp in enumerate(args): # 項でfor
        exp.backward()
        for v in exp.variables:
            if v in v_grads.keys():
                v_grads[v].append(v.grad)
            else:
                v_grads[v] = [v.grad]

    for var, grads in v_grads.items():
        total = grads[0]
        for i in range(len(grads) - 1):
            total += grads[1+i]
        if total is None:
            pass
        else:
            var.grad = mul2grad(total)

def _SumBackward(tensor):
    def _sum_(g, args, variables=[], tensor_self=None):
        v_grads = {}
        for i, exp in enumerate(args):
            exp.backward()
            grads1 = []
            for v in exp.variables:
                if v in v_grads.keys():
                    v_grads[v].append(v.grad)
                else:
                    v_grads[v] = [v.grad]

        for var, grads in v_grads.items():
            total = grads[0]
            for i in range(len(grads) - 1):
                total += grads[1+i]

            if total is not None:
                if total.data is None:
                    var.grad = wf.Tensor(np.sum(total.detach()), device=tensor.device, requires_grad=False, is_constant=False)
                else:
                    var.grad = (total * len(tensor)).no_grad()
    return _sum_

def MulBackward(*args, variables=[], wrap_tensor=None, div_grad=False):
    # len(variables) must be 2
    x = variables[0]
    y = variables[1]

    t = args[0]

    if x.is_constant and y.is_constant:
        x.grad = t / x
        y.grad = t / y
    else:
        constant_variable_list = []
        v_grads = []

        for v in variables:
            for va in v.variables:
                constant_variable_list.append(va)

        x_grads = [0.] * len(constant_variable_list)
        y_grads = [0.] * len(constant_variable_list)

        x.backward()
        for i, v in enumerate(constant_variable_list):
            x_grads[i] = v.grad if v.grad is not None else wf.Tensor(0.)

        # reset grads
        [v.zero_grad() for v in constant_variable_list]

        y.backward()
        for i, v in enumerate(constant_variable_list):
            y_grads[i] = v.grad if v.grad is not None else wf.Tensor(0.)

        [v.zero_grad() for v in constant_variable_list]

        for i, v in enumerate(constant_variable_list):
            x_grad = x_grads[i]
            y_grad = y_grads[i]

            if div_grad:
                v_grads.append((x_grad * y - x * y_grad)/(y*y))
            else:
                v_grads.append(x * y_grad + x_grad * y)

        for v, v_grad in zip(constant_variable_list, v_grads):
            v.grad = mul2grad(v_grad)

        return constant_variable_list

def DerivBackward(*args, variables=[], tensor_self=None):
    g_x = args[0] # g(x)
    d_f = args[1][0] # d_f

    if g_x is not None:
        g_x.variables = []
        for v in variables:
            if v.is_constant:
                g_x.variables.append(v)
        g_x.backward()
        for variable in variables:
            variable.grad = (variable.grad * d_f(variable)).no_grad()
    else:
        d_f(tensor_self, variables=variables) # _mul

def DivBackward(*args, variables=[], wrap_tensor=None):
    return MulBackward(*args, variables=variables, wrap_tensor=wrap_tensor, div_grad=True)

def _MeanBackward(tensor):
    def MeanBackward(*args, variables=[], wrap_tensor=None):
        total = wf.Tensor(len(tensor)).no_grad()
        for var in variables:
            var.backward()
            for v in var.variables:
                if v.grad is None:
                    v.backward()
                    for l in v.variables:
                        if l.grad is not None:
                            l.grad = mul2grad(l.grad/total)
    return MeanBackward

def _ConstantBackward(tensor_base):
    def ConstantBackward(*args, variables=[]):
        for v in args:
            if v == tensor_base:
                v.grad = wf.Tensor(1., extend=tensor_base).no_grad()
            else:
                v.grad = wf.Tensor(0., extend=tensor_base).no_grad()
    return ConstantBackward
