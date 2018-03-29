
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
        import sklearn
        #xV=X
       # yV=Y
       # model = model1_info['model']

        vd = self.validation_data
        if (len(vd)==3):
            xV, yV, _ = vd
        elif (len(vd)==4):
            xV, yV, _, _ = vd
        else:
            raise Exception("Didn't expect {} values in validation data".format(len(vd)))

        # predict returns a probability
        prob = np.asarray(self.model.predict(xV))

        if (prob.ndim==2):
         #   print("Treat as binary probabilities")
            pred  = np.where(prob > 0.5, 1, 0)
            average = 'binary'

        if (prob.ndim==3):
          #  print("Treat as category probabilities")
            pred = argmax3d(prob)
            yV   = argmax3d(yV)
            # Need to average over the categories
            average = 'micro'

        # Print first 4 observations
     #  print("X: {} {}".format(xV.shape, xV[:4]))
     #  print("Probs: {} {}".format(prob.shape, prob[:4]))
     #  print("Preds: {} {}".format(pred.shape, pred[:4]))
     #  print("Actual: {} {}".format(yV.shape, yV[:4]))

        def flatten_non_zeros(A, Z):
            # Mask out where zeros in the other array
            A[Z==0] = -1
            A = A.flatten()
            return A[A != -1]

        pred_ = flatten_non_zeros(pred, xV)
        yV_   = flatten_non_zeros(yV, xV)


        # Compare as flattened list over all obs and timestamps
        precision = sklearn.metrics.precision_score(yV_, pred_, average=average)
        logs.update({'MyPrecision':precision})


def model_load(model_name_or_info, model_path = "C:\Temp", sub_folder = None, suffix=None):
    if (type(model_name_or_info) is dict):
        model_path = model_name_or_info['model_path']
        model_name = model_name_or_info['model_name']

    if sub_folder:
        model_path = "{}\\{}".format(model_path, sub_folder)
    if suffix:
        model_name = "{}_{}".format(model_name, suffix)

    filepath = "{}\\{}.hdf5".format(model_path, model_name)

    print("Loading model from {}".format(filepath))
    model = keras.models.load_model(filepath)
    # Name doesn't get saved
    model.name = model_name
    print("Loaded model {}".format(model.name))

    if (type(model_name_or_info) is dict):
        model_name_or_info['model'] = model

    return model


def model_save(model_name_or_info, model_path="C:\Temp", model_name=None, suffix=None, echo=False, temp=True):
    if (type(model_name_or_info) is dict):
        model_path = model_name_or_info['path']
        model_name = model_name_or_info['name']
        model      = model_name_or_info['model']
    else:
        model      = model_name_or_info
        if model_name is None:
            model_name = model.name

    if temp:
        model_path = "{}\\{}".format(model_path, "Temp")
    if suffix:
        model_name = "{}_{}".format(model_name, suffix)

    filepath = "{}\\{}.hdf5".format(model_path, model_name)

    if echo: print ("Saving model to {}".format(filepath))
    model.save(filepath)






def predict_stateful(model, X, obs, n_time, n_cats):
    prob = np.zeros((n_time, n_cats))

    # Reset to run prediction for next obs - state builds up over timesteps
    model.reset_states()
    for t in range(n_time):
        Xij = np.expand_dims(np.expand_dims(X[obs][t], 1), 1)
        prob[t, :] = model.predict_on_batch(Xij)

    return prob

def test_stateful(model, X, obs, n_time, n_cats):
    loss = np.zeros(n_time)
    acc  = np.zeros(n_time)

    # Reset to run prediction for next obs - state builds up over timesteps
    model.reset_states()
    for t in range(n_time):
        Xij = np.expand_dims(np.expand_dims(X[obs][t], 1), 1)
        yij = np.expand_dims(y[obs][t], axis=0)
        loss[t], acc[t] = model.test_on_batch(Xij, yij)

    # Non-stateful models just quote an overall value for the whole sequence of timesteps
    #loss = np.mean(loss)
    #acc  = np.mean(acc)

    return loss, acc

def predict_obs(model, X, i, n_time, n_cats, stateful=False, mask_zero=True):
    if stateful:
        # Find number of populated timesteps for this obs
        n_time_i = np.count_nonzero(X[i]) if mask_zero else n_time
        # Probabilities for each category for each timestep, e.g. (10 timesteps, 28 cats)
        prob_i = predict_stateful(model, X, i, n_time=n_time_i, n_cats=n_cats)
    else:
        Xi = X[i:i + 1, ...]
        prob_i = np.squeeze(model.predict(Xi, batch_size=1, verbose=0))
    return prob_i

def evaluate_obs(model, X, y, i, n_time, n_cats, stateful=False, mask_zero=True):
    if stateful:
        # Find number of populated timesteps for this obs
        n_time_i = np.count_nonzero(X[i]) if mask_zero else n_time
        loss_i, acc_i = test_stateful(model, X, i, n_time=n_time_i, n_cats=n_cats)
        return loss_i.tolist(), acc_i.tolist()

    else:
        Xi = X[i:i + 1, ...]
        yi = y[i:i + 1, ...]
        loss_i, acc_i = model.evaluate(Xi, yi, batch_size=1, verbose=0)
        return [loss_i], [acc_i]

def pred_counts(model_or_info, X, y, n_top=3, results='s', n_obs=None, n_find=None, mask_zero=True, stateful=False, coder=None):
    '''
        Returns a summary of whether the top predictions were found in the actual subsequent timesteps
        - Rows represent which prediction it was (0=prediction with highest probability)
        - Columns show the percentage where this prediction occured:
            - 0 = not at all
            - 1 = as the next item
            - 2 = as the 2nd item...etc
        - E.g. 36.60% of the top predictions were found in the next position

          n_Top         0          1          2           3
        -------  --------  ---------  ---------  ----------
              0  0.612539  0.366063   0.0134987  0.00789921
              1  0.916608  0.0556944  0.0150985  0.0125987
              2  0.981702  0.0133987  0.0029997  0.00189981

    :param model_or_info:
    :param X:
    :param y:
    :param n_top:
    :param results: Set to 's' for Summary, 'c' for Counts, 'p' for Prefixes, 'd' for Detail, or combinations e.g. 'scd'
    :param n_obs:
    :param n_find:
    :param mask_zero:
    :param stateful:
    :param coder:
        model_info = model1_info
        model = model_info['model']
        y=Y
    :return:
    '''
    import collections
    from tabulate import tabulate as tab

    if (type(model_or_info) is dict):
        model = model_or_info['model']
    else:
        model = model_or_info

    print("Evaluating predictions for model {}".format(model.name))

    # model=model5
    details  = True if 'd' in results else False
    summary  = True if 's' in results else False
    prefixes = True if 'p' in results else False
    counts   = True if 'c' in results else False

    if n_obs is None: n_obs = X.shape[0]
    n_time = X.shape[1]
    n_cats  = y.shape[2]
    if n_find is None: n_find = n_time

    print("- based on {} obs with {} categories and upto {} timesteps ".format(n_obs, n_cats, n_time ))


    # Counts summarise for each (prefix size, prediction rank, find position)
    countsD = collections.defaultdict(lambda:0)
    detail = []
    # i=91
    #Xi = X[i,...]
    #yi = y[i,...]
    def update_for_obs(Xi, yi, prob_i, n_time_i):
        # Get actual values for all but the last timestep (which predicts a padded value)
        y_cats = np.argmax(yi[0:n_time_i-1], axis=1)
        if coder: y_cats = coder.inverse_transform(y_cats-1)
        # For each position 'j' in the time steps where a prediction is made
        # j=0
        for j in range(n_time_i):
            # Get predicted categories for this position in descending order: probSr = most likely category first in series
            if coder:
                probSr = pa.Series(data=prob_i[j,1:], index=coder.classes_).sort_values(ascending=False)
            else:
                probSr = pa.Series(data=prob_i[j]).sort_values(ascending=False)

            # Get all the remaining actual categories
            y_rest        = y_cats[j:]
            # Loop through the top predictions
            # t=2
            for t in range(n_top):
                next_top_pred = probSr.index[t]     # Predicted code (or desc if coder provided)
                next_top_prob = probSr.values[t]    # Probability of this code

                # See if the predicted code occurs in the actual values
                find_preds    = list(np.where(y_rest==next_top_pred)[0])
                # Check if it has been found, and how far ahead
                if (len(find_preds)>0):
                    # Use the first occurrence of the predicted value (index 0)
                    # f=1 means found as the expect value for this time-step (i.e. it was the next item)
                    f = find_preds[0] + 1
                    # Truncate if index where prediction is found is beyond the range
                    if (f > n_find) : f = n_find + 1
                else:
                    f = 0  # Not found

                # Always record count, so f=0 is count of when prediction was not found
                countsD[(j, t, f)] += 1

                if details:
                    # Get first part of X upto 'j' where the prediction is being made
                    if coder:
                        # Subtract 1 to allow for padding
                        x_pfx = coder.inverse_transform(Xi[0:j + 1] - 1)
                    else:
                        x_pfx = Xi[0:j + 1]

                    detail.append({'Pfx':x_pfx, 'Pred':next_top_pred, 'Prob':next_top_prob, 'n_Pfx':j, 'n_Top':t, 'n_Find':f})

    for i in range(n_obs):
        # Find number of non-zero items for this obs, or use fixed n_time if not masking zero
        n_time_i = np.count_nonzero(X[i]) if mask_zero else n_time
        # Predict probability of each category for every timestep for this obs
        prob_i   = predict_obs(model, X, i, n_time, n_cats, stateful=stateful, mask_zero=mask_zero)
        # Increment count matrix for these probabilities based on the number of non-zero inputs
        update_for_obs(X[i,...], y[i,...], prob_i, n_time_i)


    countsDf = pa.Series(countsD).reset_index()
    countsDf.columns = (['n_Pfx','n_Top', 'n_Find', 'Count'])
    summaryDf_tf = pa.pivot_table(countsDf, index='n_Top', columns='n_Find', values='Count', aggfunc=np.sum)
    summaryDf_t  = pa.pivot_table(countsDf, index='n_Top', values='Count', aggfunc=np.sum)
    # summaryDf_t gives count of predictions for each value of n_Top.
    # - All will be the same: the number of input values that were not-padded and so had a prediction
    n_pred = summaryDf_t.iloc[0, 0]
    summaryDf = summaryDf_tf / n_pred

    if (type(model_or_info) is dict):
        model_or_info.update({'summary':summaryDf})

    results = []
    if summary:
        results.append(summaryDf)
        print(tab(summaryDf, headers='keys'))
        print()

    if counts: results.append(countsDf)

    if prefixes:
        summaryDf2 = pa.pivot_table(countsDf, index=['n_Pfx','n_Top'], columns='n_Find', values='Count', aggfunc=np.sum)
        pfxDf = pa.pivot_table(countsDf, index=['n_Pfx'], values='Count', aggfunc=np.sum)
        pfxDf = summaryDf2.join(pfxDf)
        for f in range(n_find):
            pfxDf["Pct{}".format(f)] = pfxDf[f] / pfxDf['Count']
        results.append(pfxDf.reset_index())

    if details:
        detailDf = pa.DataFrame.from_dict(detail)[['Pfx', 'Pred', 'Prob', 'n_Pfx', 'n_Top', 'n_Find']]
        results.append(detailDf)

    if (len(results)==1):
        return results[0]
    else:
        return tuple(results)

