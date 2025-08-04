# -*- coding: utf-8 -*-
# Adapted by Luis Jimenez, Juan Fonseca (2017)
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_model_a.py
# 
# Changes:
# * July 2025 by Juan Fonseca. Migrated code from Keras (standalone) to the new API of Tensorflow.
#
# ** Note: **
# The code of this specific file is not ours, it was taken from Adrian Rosebrock's blog
# at PyImageSearch (August 20116). URL: http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/.
#

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Flatten
from proyecto3_utils import *

def create_model_a(input_shape, nb_classes):
	'''
	Lecun structure (LeNet-5) proposed by Ghaffari-Sharifian.
	'''
	model = Sequential()

	# layer 1
	model.add(Convolution2D(
		20, # number of convolution filters to use
		(5,5), # number of rows, columns in each convolution kernel
		input_shape=input_shape,
		name = 'conv1'
	))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	# layer 2
	model.add(Convolution2D(
		50, # number of convolution filters to use
		(5,5), # number of rows, columns in each convolution kernel
		padding="same",
		name = 'conv2'
	))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # layer 3
	model.add(Flatten())        
	model.add(Dense(500))
	model.add(Activation('relu'))

	# layer 4
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	# compile and return
	model.compile(
		loss=DEFAULT_LOSS_FUNCTION,
		optimizer=DEFAULT_OPTIMIZER,
		metrics=DEFAULT_METRICS)
	
	return model