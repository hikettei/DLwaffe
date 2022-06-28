
def cross_entropy(yt, yp):
	loss = ((yt * yp) * -1).sum()
	return loss

def mse(p, y):
	return ((p-y)**2).mean()