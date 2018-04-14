import _pickle as pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_data():
	return pickle.load(open('dataset.p','rb'))

def preprocess_game_data(game_data, inum):
	game_data = np.array(game_data)
	image_action_pairs = np.apply_along_axis(preprocess_single_sample, axis=1, arr=np.expand_dims(game_data, axis=1))
	images = image_action_pairs[:, 0]
	actions = image_action_pairs[:, 1]

	# img = Image.fromarray(images[3], 'RGB')
	# img.show()
	# img = Image.fromarray(images[4], 'RGB')
	# img.show()

	images_copy = np.copy(images)
	for i in range(1, len(images)):
		images_copy[i] = images[i] - images[i-1]
	images = images_copy
	# img = Image.fromarray(images[4], 'RGB')
	# img.show()
	# print(np.unique(images[4]))

	images_list = images.tolist()
	X = list(map(image_to_image_list, [images_list] * (len(images_list) - (inum - 1)), range(0, len(images_list) - (inum - 1)), [inum] * (len(images_list) - (inum - 1))))
	y = actions[inum - 1:].tolist()
	return np.array(X), np.expand_dims(y, axis=1)

def image_to_image_list(images, i, out_size):
	if i > len(images) - out_size:
		return
	return images[i: i + out_size]

def preprocess_single_sample(raw):
	raw = np.array(raw[0]).astype(np.int)
	image = np.zeros((80, 80, 3),dtype=np.uint8)
	# background
	image[:, :, 0] = 144
	image[:, :, 1] = 72
	image[:, :, 2] = 17

	for i in range(0, 3):
		flat_image = image[:, :, i].ravel()
		flat_image[raw[:-1]] = 1
		image[:, :, i] = np.reshape(flat_image, (80, 80))

	left_paddle_indices = np.where(image[:, :10, :] == 1)
	image[left_paddle_indices[0],left_paddle_indices[1],0] = 213
	image[left_paddle_indices[0], left_paddle_indices[1], 1] = 130
	image[left_paddle_indices[0], left_paddle_indices[1], 2] = 74

	ball_indices = np.where(image[:, :70, :] == 1)
	image[ball_indices] = 236

	right_paddle_indices = np.where(image == 1)
	image[right_paddle_indices[0], right_paddle_indices[1], 0] = 1
	image[right_paddle_indices[0], right_paddle_indices[1], 1] = 186
	image[right_paddle_indices[0], right_paddle_indices[1], 2] = 92

	return image.repeat(2,axis=0).repeat(2,axis=1), raw[-1]

if __name__ == "__main__":
	data = load_data()
	inum = 5
	X, y = preprocess_game_data(data[0], inum)
	print(type(X[0,0]))
	print(X[0, 0].shape)
	print(X.shape)

	# img = Image.fromarray(X[0, 0], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1] + X[0,2], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1] + X[0,2] + X[0,3], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1] + X[0,2] + X[0,3] + X[0,4], 'RGB')
	# img.show()