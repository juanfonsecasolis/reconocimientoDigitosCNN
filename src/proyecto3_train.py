#
# -*- coding: utf-8 -*-
# 2017 L. Jimenez, J. Fonseca
# Visión por computadora, prof. P. Alvarado ITCR
# proyecto3_train.py
#
# References: 
# - https://raw.githubusercontent.com/gradientzoo/python-gradientzoo/master/examples/keras_mnist.py
# - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# - http://www.picnet.com.au/blogs/guido/post/2016/05/16/review-of-keras-deep-learning-core-layers/
# - https://keras.io/callbacks/
# - https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

from gradientzoo import KerasGradientzoo, NotFoundError
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import sys
import argparse
from keras import backend as K
from keras import callbacks
import numpy as np
import matplotlib.pylab as plt

# bypass the certificate error on the KerasGradientZoo
import os, ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Constants
batch_size = 128
nb_classes = 10
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3
ZOO_MODEL_NAME='ericflo/mnist'
K.set_image_data_format('channels_first')
C = np.zeros((10,10)) # confusion matrix
input_shape = (1, img_rows, img_cols)

# Control flags 
DEBUG = False

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def prepareTrainingData():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	if DEBUG:
		print('X_train shape:', X_train.shape)
		print(X_train.shape[0], 'train samples')
		print(X_test.shape[0], 'test samples')
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return X_train, X_test, Y_train, Y_test

def createModelA():
        '''
	  Lecun structure (LeNet-5)
	  Code: http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
	  Mentioned by:  Ghaffari-Sharifian structure
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
              	loss='categorical_crossentropy',
                optimizer='adadelta', # an adaptive learning rate method
                metrics=['accuracy'])
        return model

def createModelC():
	'''
	Taken from: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
	'''
	model = Sequential()
	model.add(Convolution2D(
		nb_filters, # number of convolution filters to use
		(nb_conv, nb_conv), # number of rows, columns in each convolution kernel
        	padding='valid',
        	input_shape=input_shape,
		name = 'conv1'
	))
	model.add(Activation('relu'))
	model.add(Convolution2D(
		nb_filters, 
		(nb_conv, nb_conv),
		name = 'conv2'
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
		loss='categorical_crossentropy', # used to detect category labels
 		#loss='mean_squared_error',
		optimizer='adadelta', # an adaptive learning rate method
  		metrics=['accuracy'])
	return model

def createModelB():
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
                loss='categorical_crossentropy', # used to detect category labels
                optimizer='adadelta', # an adaptive learning rate method
                metrics=['accuracy'])
        return model	

def loadZooWeights(model):
	# Blocker: https://github.com/gradientzoo/python-gradientzoo/issues/1
	zoo = KerasGradientzoo(ZOO_MODEL_NAME)
	try:
		zoo.load(model)
		if DEBUG:
			print('Found and loaded latest model weights from gradientzoo')
	except NotFoundError:
		print('No existing model weights found, training for the first time')
	except Exception, e:
		if 'You are trying to load' in str(e):
		        if raw_input('Invalid weights found, continue? ')[0] != 'y':	
				sys.exit(1)
    		else:
		        raise e
	return zoo

def trainWeights(model, X_train, X_test, Y_train, Y_test, nEpochs, myCallbacks):
	model.fit(X_train, Y_train, batch_size=batch_size, epochs=nEpochs,
        	verbose=1, validation_data=(X_test, Y_test),
        	callbacks=myCallbacks)

def saveModel(model, modelpath, weightspath):
	model_json = model.to_json()
	with open(modelpath, "w") as json_file:
		json_file.write(model_json)

		# serialize weights to HDF5
		model.save_weights(weightspath)
		if DEBUG:
			print("Saved model to disk")

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--nEpochs', action='store', type=int,
                help='Number of epochs to re-train the model',default=0)
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='Output JSON file to store the trained model',default='model.json')
	parser.add_argument('-w', '--weights', action='store', type=str, 
                help='Output H5 file to store the trained weights',default='model.h5') 
	parser.add_argument('-t', '--modelType', action='store', type=str,
                help='We implemented two model structures, you can specify: A (Lecun-Bottou)  or B (Cirstea-Likforman)',default='A')
	parser.add_argument('-z', '--useZoo', action='store', type=bool,
		help='True if you want to use the zoo to speedup the training stage, False else',default=False)

	args = parser.parse_args()
	
	return args.model, args.weights, args.nEpochs, args.modelType, args.useZoo

def main():
	# read argument choices
	[modelpath, weightspath, nEpochs, modelType, useZoo] = parseArgs()

	# choose the model
	if 'A'==modelType:
		model = createModelA()
	elif 'B'==modelType:
		model = createModelB()
	else:
		print 'Error: you specified an unexistent model type'
		exit()

	# prepare training data
	[X_train, X_test, Y_train, Y_test] = prepareTrainingData()

	# initialize callbacks
	callbacks = [LossHistory()]
	if useZoo:
		zoo = loadZooWeights(model)		
		callbacks.append(zoo.make_callback(model, after_epochs=1))

	# train model and plot error
	trainWeights(model, X_train, X_test, Y_train, Y_test, nEpochs, callbacks) 
        plt.plot(callbacks[0].losses)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.show()

	# save model
	saveModel(model, modelpath, weightspath)

'''
main
'''
if '__main__' == __name__:
	main()


