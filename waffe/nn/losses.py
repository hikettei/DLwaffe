
def cross_entropy(yt, yp):
	loss = (yt * yp).sum() * -1
	return loss

def mse(p, y):
	return ((p-y)**2).mean()