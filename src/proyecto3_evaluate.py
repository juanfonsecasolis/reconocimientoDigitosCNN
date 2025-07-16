# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_evaluate.py
# 
# Referencias: 
# - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# - https://keras.io/models/model/

from proyecto3_train import prepare_training_data
from src.proyecto3_utils import *
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time
 
#K.set_image_data_format('channels_first')

def try_load(filepath):
	try:
		fileOpened = open(filepath, 'r')
	except IOError:
		raise Exception('Path unexistent: "'+filepath + '".')
	return fileOpened

def load_model(modelpath, weightspath):
	
	# load YAML and create model
	json_file = try_load(modelpath)
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	# load weights into new model
	model.load_weights(weightspath)
	if DEFAULT_DEBUG_FLAG:
		print("Loaded model from disk")
	return model

def get_train_images_mnist(i,X_test,Y_test):

	# one channel, 28x28 pixel images
	B, A = [], []
	A.append(np.array([X_test[i]]))
	B.append(np.array([Y_test[i]]))
	if DEFAULT_DEBUG_FLAG:
		print_image(A[0][0][0])
	return B, A

def print_image(img):
	plt.imshow(img)
	plt.show()
	
def load_image(imagepath):
	im = Image.open(imagepath)
	A = np.array(im)
	A = np.array([np.array([A])])
	if DEFAULT_DEBUG_FLAG:
		print_image(im)
	return A

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='JSON file containing the pretrained model',default=DEFAULT_MODEL_FILEPATH)
	parser.add_argument('-w', '--weights', action='store', type=str, 
        help='H5 file containing the pretrained weights',default=DEFAULT_H5_FILEPATH) 
	parser.add_argument('-i', '--imagepath', action='store', type=str,
        help='Path to the 28x28 grayscale BMP to analize, if none is specified then a benchmark is executed', default='none')

	args = parser.parse_args()
	
	return args.model, args.weights, args.imagepath

def execute_benchmark(model, X_test, Y_test):
	times = []
	I = 100
	TP = 0
	for i in range(I):
		[desired,image] = get_train_images_mnist(i,X_test,Y_test)
		[elapsed, obtained] = predict(model, image, False)
		times.append(elapsed)
		if(np.sum(np.subtract(desired, obtained)) < 0.5):
			TP += 1 
	print('Accuracy: %f' % (TP/I))
	print('Averaged time (%i samples): %f (ms)' % (I, np.mean(times)))

def predict(model, image):
	start = int(round(time.time() * 1000))
	result = model.predict(image)
	end = int(round(time.time() * 1000))
	elapsed = end-start

	if DEFAULT_DEBUG_FLAG:
		print('Output:')
		print(result)
		print('Classification time: %d (ms)' % elapsed)

	return elapsed, result 

'''
Main
'''
if '__main__' == __name__:

	[X_train, X_test, Y_train, Y_test] = prepare_training_data()
	[model_path, weights_path, imagepath] = parse_args()
	model = load_model(model_path, weights_path)

	# evaluate loaded model on test data
	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	if('none'==imagepath):
		print('Executing benchmark...')
		execute_benchmark(model, X_test, Y_test)
	else:
		A = load_image(imagepath)	
		predict(model, A)