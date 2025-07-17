#!/bin/bash
KERAS_BACKEND=theano
cd src
python3 proyecto3_train.py -n 1 -t B
# python3 proyecto3_plot.py
cd ..