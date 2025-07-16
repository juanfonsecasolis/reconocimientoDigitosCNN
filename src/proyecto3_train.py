# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_train.py
#
# References: 
# - https://raw.githubusercontent.com/gradientzoo/python-gradientzoo/master/examples/keras_mnist.py
# - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# - http://www.picnet.com.au/blogs/guido/post/2016/05/16/review-of-keras-deep-learning-core-layers/
# - https://keras.io/callbacks/
# - https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

from proyecto3_utils import *
from proyecto3_model_a import create_model_a
from proyecto3_model_b import create_model_b
from proyecto3_model_c import create_model_c
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K, callbacks
import argparse
import numpy as np
import matplotlib.pylab as plt

# Constants
batch_size = 128
nb_classes = 10
img_rows, img_cols, num_channels = 28, 28, 1 # dimensiones and number of channels of the image
nb_filters = 32
nb_pool = 2
nb_conv = 3
C = np.zeros((10,10)) # confusion matrix

print('Image data format: '+ K.image_data_format())
if K.image_data_format() == "channels_first":
	input_shape = (num_channels, img_rows, img_cols)
else:
	input_shape = (img_rows, img_cols, num_channels)

class loss_history(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def prepare_training_data():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	num_train_images = X_train.shape[0]
	num_test_images = X_test.shape[0]

	if K.image_data_format() == "channels_first":
		X_train = X_train.reshape(num_train_images, num_channels, img_rows, img_cols)
		X_test = X_test.reshape(num_test_images, num_channels, img_rows, img_cols)
	else:
		X_train = X_train.reshape(num_train_images, img_rows, img_cols, num_channels)
		X_test = X_test.reshape(num_test_images, img_rows, img_cols, num_channels)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	
	if DEFAULT_DEBUG_FLAG:
		print('X_train shape:', X_train.shape)
		print(num_train_images, 'train samples.')
		print(num_test_images, 'test samples.')

	Y_train = to_categorical(y_train, nb_classes)
	Y_test = to_categorical(y_test, nb_classes)

	return X_train, X_test, Y_train, Y_test

def train_weights(model, X_train, X_test, Y_train, Y_test, nEpochs, myCallbacks):
	model.fit(
		X_train, 
		Y_train, 
		batch_size=batch_size, 
		epochs=nEpochs,
        verbose=1, 
		validation_data=(X_test, Y_test),
        callbacks=myCallbacks)

def save_model(model, modelpath, weightspath):
	model_json = model.to_json()
	with open(modelpath, "w") as json_file:
		json_file.write(model_json)

		# serialize weights to HDF5
		model.save_weights(weightspath)

		if DEFAULT_DEBUG_FLAG:
			print("Saved model to disk")

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--nEpochs', action='store', type=int,
        help='Number of epochs to re-train the model',default=0)
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='Output JSON file to store the trained model',default=DEFAULT_MODEL_FILEPATH)
	parser.add_argument('-w', '--weights', action='store', type=str, 
        help='Output H5 file to store the trained weights',default=DEFAULT_H5_FILEPATH) 
	parser.add_argument('-t', '--modelType', action='store', type=str,
        help='We implemented two model structures, you can specify: A (Lecun-Bottou) or B (Cirstea-Likforman)',default='A')

	args = parser.parse_args()
	
	return args.model, args.weights, args.nEpochs, args.modelType

'''
main
'''
if '__main__' == __name__:

	# read argument choices
	[model_path, weights_path, num_epochs, model_type] = parse_args()

	# choose the model
	if 'A'==model_type:
		model = create_model_a(input_shape, nb_classes)
	elif 'B'==model_type:
		model = create_model_b(input_shape, nb_classes)
	elif 'C'==model_type:
		model = create_model_c(input_shape, nb_classes, nb_filters, nb_conv, nb_pool)
	else:
		raise Exception('You choose an unexistent model.')

	# prepare training data
	[X_train, X_test, Y_train, Y_test] = prepare_training_data()

	# initialize callbacks
	callbacks = [loss_history()]

	# train model and plot error
	train_weights(model, X_train, X_test, Y_train, Y_test, num_epochs, callbacks) 
	plt.plot(callbacks[0].losses)
	plt.xlabel('Batch')
	plt.ylabel('Loss')
	plt.title('Training result')
	plt.save(DEFAULT_OUTPUT_DIRECTORY+'/training.png')

	# save model
	save_model(model, model_path, weights_path)