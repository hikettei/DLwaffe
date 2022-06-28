
def linear(x, weight, bias):
	#Applies a linear transformation to the incoming data: y = xA^T + b
	#x      : (*, in_features)
	#weight : (out_features, in_features)
	#bias   : (out_features, 1) or None

	if bias is None:
		return x @ weight.transpose()
	else:
		return x @ weight.transpose() + bias