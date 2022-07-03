from tqdm import tqdm

class Trainer():
    def __init__(self):
        pass

def train(trainer, dataset, epoch_num=1, save_model_each=None, verbose=True):
	for epoch in range(epoch_num):
		losses = 0.
		loss = 0.
		if verbose:
			print("|=={} th Epoch============|".format(epoch))
			print("")
			batch_bar = tqdm(total=round(dataset.total()/dataset.batch_size)+1)
			batch_bar.set_description("Loss: {}".format(loss))
		while dataset.is_next():
			input_data = dataset.on_batch()
			loss = trainer.on_batch(input_data)
			losses += loss
			if verbose:
				batch_bar.update(1)
				batch_bar.set_description("Loss: {}".format(loss))
		if verbose:
			batch_bar.update(1)
			batch_bar.set_description("Loss: {}".format(losses/dataset.batch_size))
		dataset.reset()
	return trainer

def predict(trainer, dataset):
	predicts = []
	while dataset.is_next():
		input_data = dataset.on_batch()
		predicts += trainer.predict(input_data)
	dataset.reset()
	return predicts