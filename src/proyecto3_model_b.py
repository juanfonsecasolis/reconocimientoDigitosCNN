# -*- coding: utf-8 -*-
# Adapted by Luis Jimenez, Juan Fonseca (2017)
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_model_b.py
# 
# Changes:
# * July 2025 by Juan Fonseca. Migrated code from Keras (standalone) to the new API of Tensorflow.
#
# ** Note: **
# This code is an adaptation made by us of Adrian Rosebrock's code (PyImageSearch August 20116, URL: http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)
# to use the Cîrstea-Likforman structure.
#

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from proyecto3_utils import *

def create_model_b(input_shape, nb_classes):
	'''
	Cîrstea-Likforman structure
	'''
	model = Sequential()

	# layer 1
	model.add(Convolution2D(
		32, # number of convolution filters to use
		(3,3), # number of rows, columns in each convolution kernel
		input_shape=input_shape,
		name = 'conv1'
	))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	# layer 2
	model.add(Convolution2D(
		64, # number of convolution filters to use
		(3,3), # number of rows, columns in each convolution kernel
		name = 'conv2'
	))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	# layer 3
	model.add(Convolution2D(
		128, # number of convolution filters to use
		(3,3), # number of rows, columns in each convolution kernel
		name = 'conv3'
	))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())

	# fully connected layer
	model.add(Dense(625))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	# softmax layer
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	# compile and return
	model.compile(
		loss=DEFAULT_LOSS_FUNCTION,
		optimizer=DEFAULT_OPTIMIZER,
		metrics=DEFAULT_METRICS)
	
	return model