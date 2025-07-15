#!/bin/bash
KERAS_BACKEND=theano
python3 proyecto3_train.py -m ../models/model.json -w model.h5 -n 7 -t B
python3 proyecto3_plot.py -m ../models/model.json