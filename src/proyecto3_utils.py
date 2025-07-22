# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: utils.py

import keras

IMG_ROWS, IMG_COLS, NUM_CHANNELS = 28, 28, 1 # dimensions and number of channels of the input images

DEFAULT_OUTPUT_DIRECTORY = '../output'
DEFAULT_MODEL_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/model.keras'
DEFAULT_WEIGHTS_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/model.weights.h5'
DIAGRAM_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/model_diagram.png'
DEFAULT_INPUT_IMG = '../data/cinco.png'
DEFAULT_EVALUATION_RESULT_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/evaluate_model.png'
DEFAULT_TRAINING_RESULT_FILEPATH = DEFAULT_OUTPUT_DIRECTORY+'/training_result.png'
DEFAULT_METRICS_FILEPATH = DEFAULT_OUTPUT_DIRECTORY+'/metrics.txt'
DEFAULT_DEBUG_FLAG = True

# DEFAULT_OPTIMIZER = keras.optimizers.Adadelta(0.001) # an adaptive learning rate method
DEFAULT_OPTIMIZER = keras.optimizers.Adam(0.001)

DEFAULT_LOSS_FUNCTION = 'categorical_crossentropy'
# DEFAULT_LOSS_METRIC = 'mean_squared_error'

DEFAULT_METRICS = ['accuracy']