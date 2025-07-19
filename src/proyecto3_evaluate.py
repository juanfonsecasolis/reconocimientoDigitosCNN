# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_evaluate.py
# 
# Referencias: 
# - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# - https://keras.io/models/model/

from proyecto3_train import get_formated_mnist_data, format_data
from proyecto3_utils import *
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time
	
def load_image(imagepath):
	im = Image.open(imagepath)
	A = np.array(im)
	A = np.array([np.array([A])])
	return A

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='JSON file containing the pretrained model',default=DEFAULT_MODEL_FILEPATH)
	parser.add_argument('-w', '--weights', action='store', type=str, 
        help='H5 file containing the pretrained weights',default=DEFAULT_WEIGHTS_FILEPATH) 
	parser.add_argument('-i', '--imagepath', action='store', type=str,
        help='Path to the 28x28 grayscale BMP to analize, if none is specified then a benchmark is executed', default='none')

	args = parser.parse_args()
	
	return args.model, args.weights, args.imagepath

def execute_benchmark(model, X_test, Y_test):
	times = []
	I = 100
	TP = 0
	for i in range(I):
		[desired, image] = [ Y_test[i], X_test[i] ]
		print('Desired value: ' + desired)
		[elapsed, obtained] = model.predict(image, False)
		print('Obtained value: ' + obtained)

		times.append(elapsed)
		if(np.sum(np.subtract(desired, obtained)) < 0.5):
			TP += 1 
		
	print('Accuracy: %f' % (TP/I))
	print('Averaged time (%i samples): %f (ms)' % (I, np.mean(times)))

def timed_predict(model, image):
	start = int(round(time.time() * 1000))
	result = model.predict(image)
	end = int(round(time.time() * 1000))
	elapsed = end-start

	if DEFAULT_DEBUG_FLAG:
		print('Output:')
		print(result)
		print('Identified digit: ' + str(np.argmax(result)))
		print('Classification time: %d (ms)' % elapsed)
		plt.subplot(121)
		plt.stem(result[0])
		plt.xlabel('Digit')
		plt.ylabel('Certainty')
		plt.title('Classification')
		plt.subplot(122)
		plt.imshow(image[0], cmap='Greys')
		plt.savefig(DEFAULT_EVALUATION_RESULT_FILEPATH)

	return elapsed, result 

'''
Main
'''
if '__main__' == __name__:

	[X_train, X_test, Y_train, Y_test] = get_formated_mnist_data()
	[model_path, weights_path, image_filepath] = parse_args()
	model = keras.models.load_model(model_path)

	# evaluate loaded model on test data
	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	if('none'==image_filepath):
		print('Executing benchmark...')
		execute_benchmark(model, X_test, Y_test)
	else:
		unknown_image, _ = format_data(
			np.array(load_image(image_filepath)), 
			None,
			normalize_data = False)
		timed_predict(model, unknown_image)