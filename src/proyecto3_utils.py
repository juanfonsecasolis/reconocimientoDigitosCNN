# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: utils.py

DEFAULT_OUTPUT_DIRECTORY = '../output'
DEFAULT_MODEL_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/model.keras'
DEFAULT_WEIGHTS_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/model.weights.h5'
DIAGRAM_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/model.png'
DEFAULT_INPUT_IMG = '../data/cinco.png'
DEFAULT_EVALUATION_RESULT_FILEPATH = DEFAULT_OUTPUT_DIRECTORY + '/evaluate_model.png'
DEFAULT_DEBUG_FLAG = True

DEFAULT_OPTIMIZER = 'adadelta' # an adaptive learning rate method
# DEFAULT_OPTIMIZER = 'adams'

DEFAULT_LOSS_FUNCTION = 'categorical_crossentropy'
# DEFAULT_LOSS_METRIC = 'mean_squared_error'

DEFAULT_METRICS = ['accuracy']