# -*- coding: utf-8 -*-
# Authors: 2017 Luis Jimenez, Juan Fonseca
# Course: Visión por computadora, prof. P. Alvarado ITCR
# File: proyecto3_plot.py
#
# References: 
# - https://keras.io/visualization/
# - https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# - https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
# - https://keras.io/layers/convolutional/
# - https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py
# - https://github.com/fchollet/keras/issues/2733

from proyecto3_evaluate import *
from proyecto3_train import *
from proyecto3_evaluate import get_train_images_mnist
from proyecto3_train import get_formated_mnist_data
from proyecto3_utils import *
import keras
import numpy as np
import matplotlib.pyplot as plt

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = np.ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

if __name__ == '__main__':

    # 1) plot layers
    model = keras.models.load_model(DEFAULT_MODEL_FILEPATH)
    keras.utils.plot_model(model, to_file=DIAGRAM_FILEPATH)

    # 2) plot convolutional results
    iLayer = 0
    cLayer = model.layers[iLayer]
    # W = cLayer.kernel.get_value(borrow=True)
    W = cLayer.kernel._value
    print("W shape : ", W.shape)
    W = np.squeeze(W)
    print("W shape : ", W.shape)
    title = 'Layer (#'+str(iLayer)+'): '+cLayer.name
    plt.title(title)
    W2 = []
    kernel_size = cLayer.kernel_size
    nFilters = cLayer.filters
    for i in range(nFilters):
        wAct = np.zeros(kernel_size)
    for j in range(kernel_size[0]):
        for k in range(kernel_size[1]):
            wAct[j,k] = W[j,k,i]
    W2.append(wAct)

    W2 = np.array(W2)
    mosaic_x = int(np.ceil(np.sqrt(nFilters)))
    mosaic_y = int(np.ceil(np.sqrt(nFilters)))
    print("Mosaic: %s,%s" % (mosaic_x, mosaic_y))
    plt.imshow(make_mosaic(W2,mosaic_x,mosaic_y))
    plt.show()

    # 3) Visualize convolution result (after activation)
    [X_train, X_test, Y_train, Y_test] = get_formated_mnist_data()
    [Y, X] = [ Y_test[15], X_test[15]]

    # convout1_f = keras.backend.function([model.layers[0].input], model.layers[1].output)
    convout1_f = model.predict([model.layers[0].input], model.layers[1].output)
    C1 = convout1_f(X)
    C1 = np.squeeze(C1)
    print("C1 shape : ", C1.shape)
    plt.title(title)
    plt.imshow(make_mosaic(C1, mosaic_x,mosaic_y))
    plt.show()
