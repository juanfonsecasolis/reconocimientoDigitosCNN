# -*- coding: utf-8 -*-
# 2017 L. Jimenez, J. Fonseca
# Visión por computadora, prof. P. Alvarado ITCR
# proyecto3_confusionMatrix.py
# 
# Referencias: 
# - https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co

from __future__ import print_function
from src.proyecto3_evaluate import *

DEBUG = False
NUM_DIGITS_MNIST = 10
C = np.zeros((NUM_DIGITS_MNIST,NUM_DIGITS_MNIST))

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='JSON file containing the pretrained model',default='./models/model.json')
	parser.add_argument('-w', '--weights', action='store', type=str, 
                help='H5 file containing the pretrained weights',default='model.h5') 

	args = parser.parse_args()
	
	return args.model, args.weights 

def computeConfusionMatrix(model, X_test, Y_test):
	N = len(X_test)
	for i in range(N):
		[y_true, x] = getTrainImageMNIST(i,X_test,Y_test)
		y_pred = model.predict(x)
		i = int(np.argmax(y_pred))
		j = int(np.argmax(y_true))
		C[i][j] += 1

	# print confusion matrix
	print('\nConfusion Matrix',end='\n\t')
	for i in range(NUM_DIGITS_MNIST):
		print(str(i),end='\t')
	print('')
	for i in range(NUM_DIGITS_MNIST):
		print(str(i)+':',end='\t')
		for j in range(NUM_DIGITS_MNIST):
			print(C[i][j], end='\t')
		print('')
	print('\nTotal samples analyzed: ' + str(np.sum(C)),end='\n')

def computeAndPrintMetrics():
	print('\nMetrics per category:')
	TPS = []
	FPS = []
	FNS = []
	TNS = []

	minSensitivity = 999
	minSpecificity = 999
	minPrecision = 999
	maxSensitivity = 0
	maxSpecificity = 0
	maxPrecision = 0


	for i in range(NUM_DIGITS_MNIST): 	
		print('* Category '+str(i)+': ',end='')
		TP = C[i][i] # TP = A[0,0] = C[i,i] 
		FP = 0
		FN = 0
		K = range(NUM_DIGITS_MNIST)
		K.remove(i)
		J = range(NUM_DIGITS_MNIST)
		J.remove(i)

		# Split the multiclass recognition problem into identification problem
		# by fixing the category in each iteration
		#		
		#        | Classified as i | Classified as not i |
		#        -----------------------------------------
		# |Is i  |       TP        |         FN          |
		# |Not i |       FP        |         TN          |
		#
		for k in K:
			FN += C[i,k] # FN = A[0,1] = sum_{k!=i}{ C[i,k] }
			FP += C[k,i] # FP = A[1,0] = sum_{k!=i}{ C[k,i] } 			
			TN = 0
			for j in J:
				TN += C[j,k] # TN = A[1,1] = sum_{k!=i}{ sum_{j!=i}{ C[j,k] } }	

		print('TP=' + str(TP) + ', ', end='')
		print('FP=' + str(FP) + ', ', end='')
		print('FN=' + str(FN) + ', ', end='')
		print('TN=' + str(TN) + ', ', end='\n')
		[sensitivity, specificity, precision] = calculateAndPrintAdvancedMetrics(TP,FP,TN,FN)
		TPS.append(TP)
		FPS.append(FP)
		FNS.append(FN)
		TNS.append(TN)
		
		minSensitivity = min(minSensitivity, sensitivity)
		minSpecificity = min(minSpecificity, specificity)
		minPrecision = min(minPrecision, precision)
		maxSensitivity = max(maxSensitivity, sensitivity)
		maxSpecificity = max(maxSpecificity, specificity)
		maxPrecision = max(maxPrecision, precision)

	mTP = np.mean(TPS)
	mFP = np.mean(FPS)
	mTN = np.mean(TNS)
	mFN = np.mean(FNS)

	print('\nAverage:')
	print('TP=' + str(mTP) + ', ', end='')
	print('FP=' + str(mFP) + ', ', end='')
	print('TN=' + str(mTN) + ', ', end='')
	print('FN=' + str(mFN) + ', ', end='')
	print('')
	calculateAndPrintAdvancedMetrics(mTP, mFP, mTN, mFN)

	print('Min/max')
	print('Min sensitivity = ' + str(minSensitivity))
	print('Min specificity = ' + str(minSpecificity))
	print('Min precision = ' + str(minPrecision))
	print('Max sensitivity = ' + str(maxSensitivity))
	print('Max specificity = ' + str(maxSpecificity))
	print('Max precision = ' + str(maxPrecision))

def calculateAndPrintAdvancedMetrics(TP,FP,TN,FN):
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	precision = TP/(TP+FP)
	print('Sensitivity = '+str(sensitivity))
	print('Specificity = '+str(specificity))
	print('Precision = '+str(precision))
	print('')
	return sensitivity, specificity, precision

def main():
	[X_train, X_test, Y_train, Y_test] = prepareTrainingData()
	[modelpath, weightspath] = parseArgs()
	model = loadModel(modelpath, weightspath)

	# evaluate loaded model on test data
	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
	metrics=['accuracy'])

	# compute the confusion matrix and associated metrics
	computeConfusionMatrix(model, X_test, Y_test)
	computeAndPrintMetrics()

'''
Main
'''
if '__main__' == __name__:
	main()
