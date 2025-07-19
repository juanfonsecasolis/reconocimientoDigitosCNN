from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from proyecto3_utils import *

def create_model_c(input_shape, nb_classes, nb_filters, nb_conv, nb_pool):
	'''
	This code is not ours, it was taken from: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
	'''
	model = Sequential()
	model.add(Convolution2D(
		nb_filters, # number of convolution filters to use
		(nb_conv, nb_conv), # number of rows, columns in each convolution kernel
        padding='valid',
        input_shape=input_shape,
		name = 'conv1',
		data_format="channels_last"
	))
	model.add(Activation('relu'))
	model.add(Convolution2D(
		nb_filters, 
		(nb_conv, nb_conv),
		name = 'conv2',
		data_format="channels_last"
	))
	model.add(Activation('relu'))
	
	# MaxPooling: reduce the number of parameter in the next layer by keeping the 2 
	# maximum values of each filter in the current layer
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	
	# Dropout: this avoids overfitting by disabling cerain networks during the training
	model.add(Dropout(0.25))
	
	# Flatten: this flats a 2D input into a 1D array
	model.add(Flatten())
	
	# Dense: this creates a layer where each unit is fully connected to the next layer
	# (has entries to all the nodes in the next layer), the parameter is the output size of
	# the layer
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	
	# compile and return
	model.compile(
		loss=DEFAULT_LOSS_FUNCTION,
		optimizer=DEFAULT_OPTIMIZER, 
  		metrics=DEFAULT_METRICS)
	
	return model