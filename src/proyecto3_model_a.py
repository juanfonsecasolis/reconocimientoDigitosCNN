from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

def create_model_a(input_shape, nb_classes):
	'''
	Lecun structure (LeNet-5)
	This code is not ours, it was taken from: http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
	Mentioned by: Ghaffari-Sharifian structure
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