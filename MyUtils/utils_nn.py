
import pandas as pa
import numpy as np
from numpy import argmax
from sklearn.metrics import confusion_matrix
import sklearn
import MyUtils.utils_plot as myPlot
import MyUtils.utils_explore as myExp

import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
import keras as keras
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def setBins(nBins, xBins=None, yBins=None):
    if xBins is None: xBins = nBins
    if yBins is None: yBins = nBins
    return xBins, yBins

def reshapeAndSplitData(X, y, reshape=None, yCat = True, printing=True):
    '''
        Reshapes the data and split into train/valid/test as 60:20:20
    :param dfX:
    :param dfY:
    :param shape:
    :param nBins:
    :param xBins:
    :param yBins:
    :param yCat:    If True the Y values are made categorical (ie one-hot)

    :return:
    '''

    if printing:
        print("X shape: {}".format(dfX.shape))
        print("Y shape: {}".format(dfY.shape))
        print()
        print("dfX : {}".format(dfX.columns))
        print()
        print("dfY")
        myExp.overview(dfY)
        myExp.detail(dfY)

    y = dfY.as_matrix()
    nMemb = y.shape[0]
    y = y.reshape(nMemb,1)

    # Flatten to one row per person
    X = dfX.as_matrix().reshape(nMemb, -1)

    # Balance yes/no samples
    sm = SMOTE(random_state=101)
    X, y = sm.fit_sample(X, y)

    if printing:
        print("Over Sampled to give: {}".format(np.unique(y, return_counts=True)))

    if len(reshape) == 2:
        n = X.shape[0]
        X = X.reshape(n, reshape[0], reshape[1], 1)

    print("Reshaped X to {}".format(X.shape))

    if yCat:
        # Just do once for y, instead of after splitting
        print("Converting Y to categorical")
        y  = keras.utils.to_categorical(y, 2)

    X_train, X_, y_train, y_         = train_test_split(X, y,   test_size=0.4, random_state=101)
    X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, test_size=0.5, random_state=101)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def splitData(X, y, split=[80, 20], printing=True):
    '''
    :param X:
    :param y:
    :param split:
    :param printing:
    :return:
    '''

    if (len(split)==3):
        pTrain = split[0] / sum(split)
        pValid = split[1] / sum(split)
        pTest  = split[2] / sum(split)
        pTest_Valid = pTest / (pTest + pValid)

        X_train, X_, y_train, y_         = train_test_split(X, y,   test_size=1-pTrain, random_state=101)
        X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, test_size=pTest_Valid, random_state=101)

        if printing: print ("Splitting {} Train:Valid:Test giving {}, {}, {}".format(str(split), len(X_train), len(X_valid), len(X_test)))
        if printing: print ("- returning (X_train,  y_train), (X_valid, y_valid), (X_test,y_test)")
        return (X_train,  y_train), (X_valid, y_valid), (X_test,y_test)
    if (len(split) == 2):
        pTest = split[1] / sum(split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pTest, random_state=101)
        if printing: print ("Splitting {} Train:Test giving {}, {}".format(str(split), len(X_train), len(X_test)))
        if printing: print ("- returning (X_train,  y_train), (X_test,y_test)")
        return (X_train,  y_train), (X_test,y_test)

def confusion(model, x_test, y_test, show=True, mType="", n_cases=2):
    """
        return y_test, y_pred

        model, X_test, y_test = model1, X_test1, y_test1
    """

    def convert_y_if_needed(y_vals, y_lbl):
        print("{} raw shape: {}".format(y_vals.shape, y_lbl))
        if (len(y_vals.shape) == 2):
            y_vals = argmax(y_vals, axis=1)
        print("   - converted to: {}".format(y_vals.shape))
        return y_vals

    y_pred  = model.predict(x_test)
    y_pred = convert_y_if_needed(y_pred, "Y Pred")
    y_test = convert_y_if_needed(y_test, "Y Test")

    cases = range(n_cases)
    cmNN = confusion_matrix(y_test, y_pred)

    acc = sklearn.metrics.accuracy_score(y_test, y_pred)      # (TP + TN) / Total
    if n_cases==2:
       rec = sklearn.metrics.recall_score(y_test, y_pred)        # TP / (TP + FN)
       pre = sklearn.metrics.precision_score(y_test, y_pred)     # TP / (TP + FP)

    accLbl = "Proportion of classifications that are correct  = (TP + TN) / Total"
    recLbl = "Proportion of relevant cases that were selected = TP / (TP + FN)"
    preLbl = "Proportion of selected cases that are relevant  = TP / (TP + FP)"
    print()
    print("Confusion Matrix for " + mType)
    print("Accuracy  = {:.2%} = {}".format(acc, accLbl))
    if n_cases == 2:
        print("Recall    = {:.2%} = {}".format(rec, recLbl))
        print("Precision = {:.2%} = {}".format(pre, preLbl))
    print()
    if show: myPlot.plot_confusion_matrix(cmNN,cases, show=show, title=mType+" Confusion Matrix")
    stats = {"Accuracy":acc, "Precision":pre, "Recall":rec, "matrix":cmNN}
    return stats

def get_layer_dict(model):
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    for (name) in list(layer_dict):  print(name)
    return layer_dict

def find_max_input(model, layer_name, units, img_shape = None, show=False, maxTries=20, nIterations=20):

    '''
    :param model:
    :param layer_name:
    :param unit_ix:     The index of the hidden unit in the given layer
    :param img_shape:  If none, then the model.input is used to determine the shape
    :return:
    '''

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    layer_dict = get_layer_dict(model)

    # Get input tensor and find its shape (drop first dimension as this relates to the no. of observations)
    input_img = model.input
    if (img_shape is None):
        input_shape = input_img.get_shape().as_list()[1:]

    print("Creating image with shape".format(img_shape))
    layer = layer_dict[layer_name]
    layer_output = layer.output
    layer_type = type(layer)
    layer_shape = layer_output.get_shape().as_list()

    def process_unit(unit_ix):
        print('Processing unit {} in layer {} with shape {} of type {}'.format(unit_ix, layer_name, layer_shape, layer_type.__name__))
        start_time = time.time()

        # we build a loss function that maximizes the activation
        if (layer_type == Conv2D):
            loss = K.mean(layer_output[:, :, :, unit_ix])
        if (layer_type == Dense):
            loss = K.mean(layer_output[:, unit_ix])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        def tryToConverge(n):
            print ("***** Try number {} *****".format(n))
            # we start from a gray image with some random noise
            input_img_data = np.random.random((1, *input_shape))
            # we run gradient ascent for 20 steps
            for i in range(nIterations):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step
                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    return None
            return input_img_data

        def keepTrying(maxTries = 20):
            nTry = 1
            while (nTry <= maxTries):
                input_img_data = tryToConverge(nTry)
                if (input_img_data is not None):
                    # Success so stop
                    return input_img_data
                nTry += 1
            return None

        max_image = keepTrying()
        end_time = time.time()
        print('Processing took %ds' % (end_time - start_time))

        # decode the resulting input image
        if max_image is not None:
            img = deprocess_image(max_image[0])
            if show: plt.imshow(img)
            return img
        else:
            print ("Unit {} failed to converge".format(unit_ix))

    images = []
    for u in units:
        img = process_unit(u)
        if (img is not None):
            images.append(img)
    return images




def argmax3d(prob3d):
    '''
        Applies argmax of category probabilities in 3D array (n_obs x n_timestamp x n_categories)
    :param prob3d:
        Takes array n_obs x n_timestamp x n_categories
    :return:
        Returns 2D array (n_obs x n_timestamp) with the max category
    '''

    # prob3d = np.array([
    #    [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
    #    [[0.5, 0.2, 0.3], [0.1, 0.2, 0.13], [0.1, 0.2, 0.3], [0.91, 0.2, 0.3]]
    # ])

    nobs  = prob3d.shape[0]
    ntime = prob3d.shape[1]
    ncats = prob3d.shape[2]
    prob2d = prob3d.reshape(nobs * ntime, ncats)
    cats1d = np.argmax(prob2d, 1)
    cats2d = cats1d.reshape(nobs, ntime)
    return cats2d


class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs):
        xV, yV, _ = self.validation_data

        # predict returns a probability
        prob = np.asarray(self.model.predict(xV))

        if (prob.ndim==2):
            #print("Treat as binary probabilities")
            cats  = np.where(prob > 0.5, 1, 0)
            average = 'binary'

        if (prob.ndim==3):
            #print("Treat as category probabilities")
            cats = argmax3d(prob)
            yV   = argmax3d(yV)
            # Need to average over the categories
            average = 'micro'

      # print("Probs: {} {}".format(prob.shape, prob[:4]))
      # print("Preds: {} {}".format(cats.shape, cats[:4]))
      # print("Actual: {} {}".format(yV.shape, yV[:4]))

        # Compare as flattened list over all obs and timestamps
        precision = sklearn.metrics.precision_score(yV.flatten(), cats.flatten(), average=average)
        logs.update({'MyPrecision':precision})
