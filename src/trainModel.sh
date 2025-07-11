#!/bin/bash
KERAS_BACKEND=theano
python proyecto3_train.py -m ../models/model.json -w model.h5 -n 7 -t B
python proyecto3_plot.py