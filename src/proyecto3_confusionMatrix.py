# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_confusionMatrix.py
# 
# Referencias:
# - https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co

from proyecto3_evaluate import *
from proyecto3_utils import *
from proyecto3_train import get_formated_mnist_data
import keras
import pandas as pd

NUM_DIGITS_MNIST = 10
confusion_matrix = np.zeros((NUM_DIGITS_MNIST,NUM_DIGITS_MNIST)) 

def log(text, end='\n'):
	with open(DEFAULT_METRICS_FILEPATH, 'a') as file:
		file.write(text + end)

def compute_confusion_matrix(model: keras.Sequential, X_test, Y_test):
	
	N = len(X_test)
	for i in range(N):
		y_true, x = Y_test[i], X_test[i]
		y_pred = model.predict(x.reshape(-1, IMG_ROWS, IMG_COLS, NUM_CHANNELS))
		i = int(np.argmax(y_pred))
		j = int(np.argmax(y_true))
		confusion_matrix[i][j] += 1

	# log confusion matrix
	# header
	#log('# Confusion Matrix')
	
	#log('  \t')
	#for i in range(NUM_DIGITS_MNIST):
	#	log(str(i)+'|\t', end='')
	#log('') # new line

	# rows
	#for i in range(NUM_DIGITS_MNIST):
	#	log(str(i)+':'+'\t', end='')
	#	for j in range(NUM_DIGITS_MNIST):
	#		log(str(confusion_matrix[i][j])+',\t', end='')
	#	log('')	# new line

	log(str(pd.DataFrame(confusion_matrix)))

	log('\nTotal samples analyzed: ' + str(np.sum(confusion_matrix)) + '\n')

def compute_and_log_metrics():
	log('# Metrics per category:')
	TPS = []
	FPS = []
	FNS = []
	TNS = []

	min_sensitivity = 999
	min_specificity = 999
	min_precision = 999
	max_sensitivity = 0
	max_specificity = 0
	max_precision = 0

	for i in range(NUM_DIGITS_MNIST): 	
		log('Category '+str(i)+': ')
		TP = confusion_matrix[i][i] # TP = A[0,0] = C[i,i] 
		FP = 0
		FN = 0
		K = list(range(NUM_DIGITS_MNIST))
		del K[i]
		J = list(range(NUM_DIGITS_MNIST))
		del J[i]

		# Split the multiclass recognition problem into identification problem
		# by fixing the category in each iteration
		#		
		#        | Classified as i | Classified as not i |
		#        -----------------------------------------
		# |Is i  |       TP        |         FN          |
		# |Not i |       FP        |         TN          |
		#
		for k in K:
			FN += confusion_matrix[i,k] # FN = A[0,1] = sum_{k!=i}{ C[i,k] }
			FP += confusion_matrix[k,i] # FP = A[1,0] = sum_{k!=i}{ C[k,i] } 			
			TN = 0
			for j in J:
				TN += confusion_matrix[j,k] # TN = A[1,1] = sum_{k!=i}{ sum_{j!=i}{ C[j,k] } }	
		
		log('TP =' + str(TP), end=', ')
		log('FP =' + str(FP), end=', ')
		log('FN =' + str(FN), end=', ')
		log('TN =' + str(TN), end=', ')
		log('')

		[sensitivity, specificity, precision] = calculate_and_log_advanced_metrics(TP,FP,TN,FN)
		TPS.append(TP)
		FPS.append(FP)
		FNS.append(FN)
		TNS.append(TN)
		
		min_sensitivity = min(min_sensitivity, sensitivity)
		min_specificity = min(min_specificity, specificity)
		min_precision = min(min_precision, precision)
		max_sensitivity = max(max_sensitivity, sensitivity)
		max_specificity = max(max_specificity, specificity)
		max_precision = max(max_precision, precision)

	mTP = np.mean(TPS)
	mFP = np.mean(FPS)
	mTN = np.mean(TNS)
	mFN = np.mean(FNS)

	log('Average:')
	log('* TP =' + str(mTP), end='')
	log('* FP =' + str(mFP), end='')
	log('* TN =' + str(mTN), end='')
	log('* FN =' + str(mFN), end='')
	log('')

	calculate_and_log_advanced_metrics(mTP, mFP, mTN, mFN)
	log('Min')
	log('* Sensitivity = ' + str(min_sensitivity))
	log('* Specificity = ' + str(min_specificity))
	log('* Precision = ' + str(min_precision))

	log('Max')
	log('* Sensitivity = ' + str(max_sensitivity))
	log('* Specificity = ' + str(max_specificity))
	log('* Precision = ' + str(max_precision))

def calculate_and_log_advanced_metrics(TP,FP,TN,FN):
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	precision = TP/(TP+FP)
	log('* Sensitivity = '+str(sensitivity))
	log('* Specificity = '+str(specificity))
	log('* Precision = '+str(precision))
	log('')
	return sensitivity, specificity, precision

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', action='store', type=str, 
		help='JSON file containing the pretrained model.',default=DEFAULT_MODEL_FILEPATH)
	return parser.parse_args().model

'''
Main
'''
if '__main__' == __name__:
	[X_train, X_test, Y_train, Y_test] = get_formated_mnist_data()
	model_path = parse_args()
	model = keras.models.load_model(model_path)

	# evaluate loaded model on test data
	model.compile(
		loss=DEFAULT_LOSS_FUNCTION,
		optimizer=DEFAULT_OPTIMIZER,
		metrics=DEFAULT_METRICS)

	# compute the confusion matrix and associated metrics
	compute_confusion_matrix(model, X_test, Y_test)
	compute_and_log_metrics()
