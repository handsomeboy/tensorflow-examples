import numpy as np

def get_batch(train_x, train_y, batch_size = 64):
	assert train_x.shape[0] == train_y.shape[0], "Houston we have a problem"
	if batch_size > train_x.shape[0]:
		print("batch size must be at most {} but is {}, defaulting to 64".format(train_x.shape[0], batch_size))
		batch_size = 64
	idx = np.random.randint(train_x.shape[0], size = batch_size) # actually gets a list to select
	batch_x = train_x[idx, :]
	batch_y = train_y[idx, :]
	return batch_x, batch_y