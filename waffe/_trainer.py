from tqdm import tqdm

class Trainer():
    def __init__(self):
        pass

def train(trainer, dataset, epoch_num=1, save_model_each=None, verbose=True):
	for epoch in range(epoch_num):
		loss = 0.

		if verbose:
			print("|=={} th Epoch============|".format(epoch))
			batch_bar = tqdm(total=round(dataset.total()/dataset.batch_size)+1)
			batch_bar.set_description("Loss: {}".format(loss))
		while dataset.is_next():
			input_data = dataset.on_batch()
			loss = trainer.on_batch(input_data)
			if verbose:
				batch_bar.update(1)
				batch_bar.set_description("Loss: {}".format(loss))
		dataset.reset()

	return trainer