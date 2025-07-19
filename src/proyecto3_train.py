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
import keras
import argparse
import numpy as np
import matplotlib.pylab as plt

# Constants
batch_size = 128
nb_classes = 10
nb_filters = 32
nb_pool = 2
nb_conv = 3
C = np.zeros((10,10)) # confusion matrix

class loss_history(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def format_data(X, Y, normalize_data = False):
	num_images = X.shape[0]

	print('Keras image format: ' + keras.backend.image_data_format())
	if keras.backend.image_data_format() == "channels_first":
		# Theano-style: NCHW (channel-first)
		X_curated = X.reshape(num_images, NUM_CHANNELS, IMG_ROWS, IMG_COLS)
	elif keras.backend.image_data_format() == "channels_last":
		# Tensorflow-style: NHWC (channel-last)
		X_curated = X.reshape(num_images, IMG_ROWS, IMG_COLS, NUM_CHANNELS)
	else:
		raise Exception('Unknown image format "' + keras.backend.image_data_format() + '".')
	
	if normalize_data:
		X_curated = X_curated.astype('float32')
		X_curated /= 255

	if DEFAULT_DEBUG_FLAG:
		print('Number of images: ' + str(num_images) + '.')
		print('Old X shape:', X.shape)
		print('New X shape:', X_curated.shape)

	if Y is None:
		Y_curated = Y
	else:
		Y_curated = keras.utils.to_categorical(Y, nb_classes)
		if DEFAULT_DEBUG_FLAG:
			print('Old Y shape:', Y.shape)
			print('New Y shape:', Y_curated.shape)

	return X_curated, Y_curated

def get_formated_mnist_data():
	(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

	X_train_curated, Y_train_curated = format_data(X_train, Y_train)
	X_test_curated, Y_test_curated = format_data(X_test, Y_test)

	return X_train_curated, X_test_curated, Y_train_curated, Y_test_curated

def train_weights(model, X_train, X_test, Y_train, Y_test, nEpochs, myCallbacks):
	model.fit(
		X_train, 
		Y_train, 
		batch_size=batch_size, 
		epochs=nEpochs,
        verbose=1, 
		validation_data=(X_test, Y_test),
        callbacks=myCallbacks)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--nEpochs', action='store', type=int,
        help='Number of epochs to re-train the model.',default=0)
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='Output JSON file to store the trained model.',default=DEFAULT_MODEL_FILEPATH) 
	parser.add_argument('-t', '--modelType', action='store', type=str,
        help='We implemented two model structures, you can specify: A (Lecun-Bottou) or B (Cirstea-Likforman).',default='A')

	args = parser.parse_args()
	return args.model, args.nEpochs, args.modelType

'''
main
'''
if '__main__' == __name__:

	# read argument choices
	[model_path, num_epochs, model_type] = parse_args()

	# determine the input shape
	print('Image data format: '+ keras.backend.image_data_format())
	if keras.backend.image_data_format() == "channels_first":
		input_shape = (NUM_CHANNELS, IMG_ROWS, IMG_COLS)
	if keras.backend.image_data_format() == "channels_last":
		input_shape = (IMG_ROWS, IMG_COLS, NUM_CHANNELS)
	else:
		raise Exception('Unknown image format "' + keras.backend.image_data_format() + '".')

	# choose the model
	if 'A'==model_type:
		model = create_model_a(input_shape, nb_classes)
	elif 'B'==model_type:
		model = create_model_b(input_shape, nb_classes)
	elif 'C'==model_type:
		model = create_model_c(input_shape, nb_classes, nb_filters, nb_conv, nb_pool)
	else:
		raise Exception('Please choose an existent model.')

	# prepare training data
	[X_train, X_test, Y_train, Y_test] = get_formated_mnist_data()

	# initialize callbacks
	callbacks = [loss_history()]

	# train model and plot error
	train_weights(model, X_train, X_test, Y_train, Y_test, num_epochs, callbacks) 
	plt.plot(callbacks[0].losses)
	plt.xlabel('Batch iteration')
	plt.ylabel('Error')
	plt.title('Training result')
	plt.savefig(DEFAULT_TRAINING_RESULT_FILEPATH)

	# save model
	model.save(model_path)