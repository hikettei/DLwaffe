
def wftensor_to_str(tensor):
    # tensor([])
    ls = ["WfTensor(", str(tensor.detach())]

    if tensor.requires_grad:
    	ls += [", grad_fn=<", tensor.backwards["grad_name"], ">"]

    ls += [")"]
    return "".join(ls)