# -*- coding: utf-8 -*-
# 2017 L. Jimenez, J. Fonseca
# Visión por computadora, prof. P. Alvarado ITCR
# proyecto3_evaluate.py
# 
# Referencias: 
# - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# - https://keras.io/models/model/

from proyecto3_train import prepareTrainingData
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from keras import backend as K
import time

DEBUG = False 
K.set_image_data_format('channels_first')

def tryLoad(filepath):
	try:
		fileOpened = open(filepath, 'r')
	except IOError:
		raise Exception('Path unexistent: "'+filepath + '".')
	return fileOpened

def loadModel(modelpath, weightspath):
	# load YAML and create model
	json_file = tryLoad(modelpath)
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(weightspath)
	if DEBUG:
		print("Loaded model from disk")
	return model

def getTrainImageMNIST(i,X_test,Y_test):
	# one channel, 28x28 pixel images
	B, A = [], []
	A.append(np.array([X_test[i]]))
	B.append(np.array([Y_test[i]]))
	if DEBUG:
		printImage(A[0][0][0])
	return B, A

def printImage(img):
	plt.imshow(img)
	plt.show()
	
def loadImage(imagepath):
	im = Image.open(imagepath)
	A = np.array(im)
	A = np.array([np.array([A])])
	if DEBUG:
		printImage(im)
	return A

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='JSON file containing the pretrained model',default='../models/model.json')
	parser.add_argument('-w', '--weights', action='store', type=str, 
        help='H5 file containing the pretrained weights',default='../models/model.h5') 
	parser.add_argument('-i', '--imagepath', action='store', type=str,
        help='Path to the 28x28 grayscale BMP to analize, if none is specified then a benchmark is executed', default='none')

	args = parser.parse_args()
	
	return args.model, args.weights, args.imagepath

def executeBenchmark(model, X_test, Y_test):
	times = []
	I = 100
	TP = 0
	for i in range(I):
		[desired,image] = getTrainImageMNIST(i,X_test,Y_test)
		[elapsed, obtained] = predict(model, image, False)
		times.append(elapsed)
		if(np.sum(np.subtract(desired, obtained)) < 0.5):
			TP += 1 
	print('Accuracy: %f' % (TP/I))
	print('Averaged time (%i samples): %f (ms)' % (I, np.mean(times)))

def predict(model, image, verbose=True):
	start = int(round(time.time() * 1000))
	result = model.predict(image)
	end = int(round(time.time() * 1000))
	elapsed = end-start
	if verbose:
		print('Output:')
		print(result)
		print('Classification time: %d (ms)' % elapsed)
	return elapsed, result 

'''
Main
'''
if '__main__' == __name__:

	[X_train, X_test, Y_train, Y_test] = prepareTrainingData()
	[modelpath, weightspath, imagepath] = parseArgs()
	model = loadModel(modelpath, weightspath)

	# evaluate loaded model on test data
	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	if('none'==imagepath):
		print('Executing benchmark...')
		executeBenchmark(model, X_test, Y_test)
	else:
	
		# one channel, 28x28 pixel images
		#[B,A] = getTrainImageMNIST(15, X_test, Y_test)
		A = loadImage(imagepath)	
		predict(model, A)