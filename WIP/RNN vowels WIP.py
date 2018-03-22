
from __future__ import print_function

import re
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Input, Flatten, Dropout, TimeDistributed, BatchNormalization, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
import keras.utils
import keras
import numpy as np
import pdb
import sklearn
from sklearn import preprocessing
import keras.preprocessing.sequence as K_prep_seq
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import pandas as pa
import MyUtils.utils_nn as MyNn
import MyUtils.utils_plot as MyPlot
import MyUtils.utils_ui as myUi
import time



## Getting warnings from sklearn doing coder.inverse_transform(2)
## DeprecationWarning: The truth value of an empty array is ambiguous.
# Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#data_path = r"C:\Temp\TestPyCharm\Data\Models\002 RNN vowel"
models_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Data\Models\002 RNN vowel"
data_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Data\Sample Data"

############# Raw Data #############

def read_file(filename=None, n=10000):
    if filename is None : filename = "english text.txt"
    filepath = data_path + "\\" + filename
    print("Reading filename: {}".format(filepath))
    with tf.gfile.GFile(filepath, "r") as f:
        text = f.read(n).replace("\n", " ")
    print("Read {} characters".format(len(text)))
    return text

def prep_text(input_text:str):
    import string

    # Each split char separated by |
    splitter = '; |, |\r|\n| '
    x = input_text.lower()
    punc   = str.maketrans('', '', string.punctuation)
    digits = str.maketrans('', '', '1234567890')
    other = str.maketrans('', '', '-_*—°×–−')
    x = x.translate(punc)
    x = x.translate(digits)
    x = x.translate(other)
    x = re.split(splitter, x)
    x = [i for i in x if i]
    return x

def words_to_raw_data(words):
    lol =  [(list(w+' ')) for w in words]
    return lol


def create_encoder(raw_data):
    '''
    :param list_of_lists:
    :return:
        the coded input, and the coder itself
        coder.transform(['e'])
        coder.inverse_transform(3)
    '''

    unique_items = set()
    for raw_obs in raw_data:
        for item in raw_obs:
            unique_items.add(item)

    coder = preprocessing.LabelEncoder()
    coder.fit(list(unique_items))
    print(coder.classes_[:30])

    return coder


def get_raw_data(n_chars = 1000, filename=None):
    text      = read_file(n=n_chars, filename=filename)
    words     = prep_text(text)
    raw_data  = words_to_raw_data(words)

    max_len   = max([len(raw_obs) for raw_obs in raw_data])
    n_obs = len(raw_data)
    print("Got {} words with max length {}".format(n_obs, max_len))

    coder = create_encoder(raw_data)
    return raw_data, coder


############ X Y Data ##############

def pad_data(data, max_len, value):
    return K_prep_seq.pad_sequences(data, maxlen=max_len, value=value, padding='post')

def encode_raw_data(coder, raw_data, max_len=None):
    #raw_obs=raw_data[0]
    coded_data = [list(coder.transform(raw_obs)) for raw_obs in raw_data]

   #'#for i,raw_obs in enumerate(raw_data):
   #    print(i)
   #    coder.transform(raw_obs)


    if max_len:
        coded_data = pad_data(coded_data, max_len, -1)
    return coded_data

def transform_x_data(coder, coded_data):
    max_code = len(coder.classes_)
    return (coded_data + 1) / max_code



def is_vowel(raw_item):
    vowels = set(list('aeiou'))
    return vowels.__contains__(raw_item)

def flag_items(raw_obs, is_flag, value=1):
    flags = [value if is_flag(i) else 0 for i in raw_obs]
    return np.array(flags)

def flag_data(raw_obs, flags):
    is_space = lambda x: x == ' '
    vowels = flag_items(raw_obs, is_vowel, value=1)
    #raw_obs = raw_data[0]
    if flags == 2:
        spaces = flag_items(raw_obs, is_space, value=2)
    if flags == 1:
        spaces=0
    return list(vowels + spaces)

def createY(raw_data, max_len = 10, n_dims=3, flags=1):
    if flags==0:
        flagged = encode_raw_data(coder, raw_data, max_len=max_len)+1
    else:
        flagged = np.array([flag_data(raw_obs, flags=flags) for raw_obs in raw_data])

    padded  = K_prep_seq.pad_sequences(flagged, maxlen = max_len, value=0, padding='post')

    # Shift to the left, so as to have the vowel status of item n+1 against item n
    shifted  = shift(padded, [0, -1], mode='constant', cval=0)

    print(shifted[:4])
    n_obs=len(raw_data)

    if flags==1:
        y = shifted
        y_shape = (n_obs, max_len, 1)[0:n_dims]
    else:
        y = keras.utils.to_categorical(shifted)
        n_cats = np.max(shifted) + 1
        y_shape = (n_obs, max_len, n_cats)

    y = y.reshape(y_shape)
    print(y.shape)

    return y

def createX(raw_data, coder, max_len = 10, normalise=True, n_dims=3, printing='True'):
    x = encode_raw_data(coder, raw_data, max_len=max_len)
    # Allow original integer values to be returned for use in embeddings
    if normalise:
        x = transform_x_data(coder, x)
    else:
        # Turns padded -1's into 0's
        x = x + 1

    if printing: print(x[:4])

    n_obs = len(raw_data)
    x_shape = (n_obs, max_len, 1)[0:n_dims]
    x = x.reshape(x_shape)

    return x

def print_sample(coded_data, coder, n=5):
    for coded_obs in coded_data[:n]:
        decode_obs = [coder.inverse_transform(code) for code in coded_obs]
        print(decode_obs)

def get_xy_data(raw_data, coder, max_len=10, normalise=True, x_dims=3, y_dims=3, flags=1):
    max_len   = min(max_len, max([len(raw_obs) for raw_obs in raw_data]))
    n_obs = len(raw_data)
    print("Applying max length {} to {} words".format(max_len, n_obs))
    print(raw_data[:4])

    X = createX(raw_data, coder, max_len = max_len, normalise=normalise, n_dims=x_dims)
    y = createY(raw_data, max_len = max_len, n_dims=y_dims, flags=flags)

    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))

    return X,y






###########  Models  #############

def create_model_A(hidden_units, input_shape, model_name="ModelA"):
    model = Sequential(name=model_name)
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def create_model_B(hidden_units, input_shape, model_name="ModelB"):
    '''
        2 layer model:
            - Input is 3D
            - Output is binary
    :param hidden_units:
    :param input_shape:
    :return:
    '''
    model = Sequential(name=model_name)
    model.add(LSTM(hidden_units[0], input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.8))

    model.add(LSTM(hidden_units[1], return_sequences=True))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.8))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


def create_model_C(hidden_units, max_len, num_categories, embedding_size, model_name="ModelC"):
    '''
        Input has to be 2 dimensions: n_obs * n_time_stamp (with no n_features)
        Output is binary
    :param hidden_units:
    :param max_len:
    :param num_categories:
    :param embedding_size:
    :return:
    '''
    model = Sequential(name=model_name)
    #input = Input(shape=input_shape, dtype='int32')
    #print (type(input))
    #model.add()

    model.add(Embedding(input_dim=num_categories, input_length=max_len, output_dim=embedding_size)) # , dropout=0.2, mask_zero=True))
   # model.add(Flatten())
   # model.add(Dense(1, activation = "sigmoid"))

    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


def create_model_D(hidden_units, max_len, num_categories, embedding_size, num_flags, mask_zero=False, model_name="ModelD"):
    '''
        Input has to be 2 dimensions: n_obs * n_time_stamp (with no n_features)
        Output is categorical
    :param hidden_units:
    :param max_len:
    :param num_categories:
    :param embedding_size:
    :return:
    '''
    model = Sequential(name=model_name)
    model.add(Embedding(input_dim=num_categories, input_length=max_len, output_dim=embedding_size, mask_zero=mask_zero)) # , dropout=0.2, mask_zero=True))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(TimeDistributed(Dense(num_flags+1, activation = "softmax")))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


def create_model_E(hidden_units, num_flags, model_name="ModelE"):
    model = Sequential(name=model_name)
    model.add(LSTM(hidden_units, batch_input_shape=(1,1,1), return_sequences=False, stateful=True))
    model.add(Dense(num_flags+1, activation = "softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def create_model_F(hidden_units, num_flags, embedding_size, time_steps=1, batch_size=1, mask_zero=True, model_name="ModelF"):
    model = Sequential(name=model_name)
    model.add(Embedding(input_dim=num_flags+1,
                        input_length=time_steps,
                        batch_input_shape=(batch_size, time_steps),
                        output_dim=embedding_size,
                        mask_zero=mask_zero))
    model.add(LSTM(hidden_units,
                   batch_input_shape=(batch_size, time_steps, embedding_size),
                   return_sequences=False, stateful=True))
    model.add(Dense(num_flags+1, activation = "softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model




#################  Process models

def model_load(model_name):
    filepath = "{}\\{}.hdf5".format(models_path, model_name)
    print("Loading model from {}".format(filepath))
    model = load_model(filepath)
    # Name doesn't get saved
    model.name = model_name
    print("Loaded model {}".format(model.name))
    return model


def model_save(model, model_path, model_name, sfx, echo=False, temp=True):
    if temp==True:
        temp_path="\\temp"
    else:
        temp_path=""
    filename = "{}{}\\model_{}_{}.hdf5".format(model_path, temp_path, model_name, sfx)
    if echo: print ("Saving model to {}".format(filename))
    model.save(filename)


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


def model_fit(model, X, y, epochs, batch_size, stateful=False, shuffle=True, save=True):
    # When running as stateful, the whole training set is the single large sequence, so must not shuffle it.
    # When not stateful, each item in the training set is a different individual sequence, so can shuffle these
    if stateful:
        shuffle = False
        batch_size = 1
        lbl = "Iteration"
        timesteps = X.shpape[1]
        if (timesteps != 1):
            raise ValueError("When using stateful it is assumed that each X value has a single time-step but there are {}".format(timesteps))
    else:
        lbl = "Epoch"

    print("Fitting model '{}' over {} epochs".format(model_name,epochs))
    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))
    print()
    #checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    metrics = Metrics()
    precision   = []
    accuracy = []
    loss = []
    for epoch in range(epochs):
        # if the shuffle argument in model.fit is set to True (which is the default),
        # the training data will be randomly shuffled at each epoch
        h = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=shuffle
                      ,validation_split=0.25
                      ,callbacks=[metrics]
                      ).history

        print("{} {:4d} : loss {:.04f}, accuracy {:0.4f}, Precision {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], h['MyPrecision'][0], time.ctime()))
        accuracy += h['acc']
        precision += h['MyPrecision']
        loss += h['loss']

        # When not stateful, state is reset automatically after each input
        # When stateful, this is suppressed, so must manually reset after the epoch (effectively the one big sequence)
        if stateful: model.reset_states()

        if save: model_save(model, models_path, model_name, "latest")

        if not (epoch % 10):
            if save: model_save(model, models_path, model_name, epoch)


    if save: model_save(model, models_path, model_name, "final", echo=True, temp=False)
    return precision

def model_fit_stateful(model, X, y, epochs, save=True, mask_zero=True, n_obs=None):
    #model = model6
    if n_obs is not None:
        X=X[0:n_obs,:,:]
        y=y[0:n_obs,:,:]

    print("Fitting model '{}' over {} epochs".format(model_name,epochs))
    print("X shape: {}".format(X.shape)) # (450, 10, 1)
    print("y shape: {}".format(y.shape)) # (450, 10, 28)
    print()
    n_obs  = X.shape[0]
    n_time = X.shape[1]
    accuracy  = []
    loss = []
    #epoch,i,j = (0,0,0)
    for epoch in range(epochs):
        tr_acc_s = []
        tr_loss_s = []
        for i in range(n_obs):
            for j in range(n_time):
                Xij = np.expand_dims(np.expand_dims(X[i][j], 1), 1)
                yi  = np.expand_dims(y[i][j], axis=0)  # (10, 28)
                if (mask_zero & (np.squeeze(Xij)==0)):
                    #print("Skipping after length {}".format(j))
                    break

                tr_loss, tr_acc = model.train_on_batch(Xij, yi)
                tr_acc_s.append(tr_acc)
                tr_loss_s.append(tr_loss)

            # Reset ready for next obs
            model.reset_states()

        accuracy.append(np.mean(tr_acc_s))
        loss.append(np.mean(tr_loss_s))

        print("Epoch {:4d} : loss {:.04f}, accuracy {:0.4f} - {}".format(epoch, loss[-1],accuracy[-1], time.ctime()))

        if save: model_save(model, models_path, model_name, "latest")
        if not (epoch % 10):
            if save: model_save(model, models_path, model_name, epoch)

    if save: model_save(model, models_path, model_name, "final", echo=True, temp=False)

    #return tr_loss_s, tr_acc_s
    return accuracy


##################  Evaluate Models #####################


def predict(model_info, word, flag=1, max_len=10, printing=False):
    raw_obs = list(word)
    model, x_dims, normalise = model_info

    x_data = createX([raw_obs], coder, max_len = max_len, normalise=normalise)
    x_shape = (1, max_len, 1)[0:x_dims]
    X = x_data.reshape(x_shape)

    if printing: print("X shape: {}".format(X.shape))

    result = model.predict(X, batch_size=1, verbose=0)

    # Remove redundant first axis
    if max_len > 1:
        # Shape is (1 obs, t time-step, c cats)
        result = np.squeeze(result, axis=0)
    # For stateful, max_len is 1, and shape is already (1 obs, c cats)

    # Keep first dimension, to get all timeframes
    # Get the prob of the flag
    if ((result.shape[1]) == 1):
        # Binary prediction is a single prob
        pred = np.squeeze(result)
    elif (flag is not None):
        # Pick all timesteps for the given flag
        pred = result[:,flag]
    else:
        #pred = np.argmax(result, axis=1)
       # print ("Returning all categories")
        # Drop first value which is for padding, second is predict no category
        probs = result[:,1:].T
        # DF of 10 timeframes and prediction of each category
        pred = pa.DataFrame(probs, coder.classes_)
    return pred


def predict_next_letter(model_info, text, ax=None, top_n=None, max_len=10, printing=False):
    '''
    :param model_info:
    :param letters:
    :param prefix:
    :param max_len:
    :param flag: Which category to give the probability for.  If None then returns the argmax category
    :return:
    '''
    if ax is None:
        ax = plt.gca()
    else:
        ax.clear()

    raw_obs = list(text)
    prediction_index = len(text)-1
    pred = predict(model_info, raw_obs, flag=None, max_len=max_len, printing=printing)

    # Keep 5 most likely next letters
    predCol = pred.columns[prediction_index]
    if top_n:
        predDf = pred.nlargest(top_n, predCol)[[predCol]]
    else:
        predDf = pred[[predCol]]
    predDf.columns = [text]
    results = predDf
    ax.bar(results.index.values, results[text])
    ax.set_title ("Probability of letter following: {}...".format(text))
    #ax.set_title ("Results for model {}".format(model.name))


def predict_next_letter_stateful(model_info, text, ax=None, top_n=None, printing=False):
 #   text = 'hello'
#    top_n = 5

    if ax is None:
        ax = plt.gca()
    else:
        ax.clear()

    model, _, _ = model_info

    # Always reset and then play in the full list of letters
    model.reset_states()
    for i, letter in enumerate(text):
        pred = predict(model_info, [letter], flag=None, max_len=1, printing=printing)


    # Keep 5 most likely next letters from the last prediction
    predCol = pred.columns[0]
    if top_n:
        predDf = pred.nlargest(top_n, predCol)[[predCol]]
    else:
        predDf = pred[[predCol]]

    predDf.columns = [text]
    results = predDf
    ax.bar(results.index.values, results[text])
    ax.set_title ("Probability of letter following: {}...".format(text))
    #ax.set_title ("Results for model {}".format(model.name))


def predict_each(model_info, letters, prefix='', flag=1, max_len=10, sorting=False):
    '''
    :param model_info:
    :param letters:
    :param prefix:
    :param max_len:
    :param sorting:
    :param flag: Which category to give the probability for.  If None then returns the argmax category
    :return:
    '''
    model, x_dims, normalise = model_info
    results=pa.DataFrame()
    prediction_index = [len(prefix)]
    for raw_item in list(letters):
        raw_obs = list(prefix) + list(raw_item)
        chars = ''.join(raw_obs)
        pred = predict(model_info, raw_obs, flag=flag, max_len=max_len)
        if flag is None:
            # Keep 5 most likely next letters
            predDf = pred.iloc[:,prediction_index].nlargest(5, pred.columns[prediction_index])
            predDf.columns = [chars]
            if (len(results)==0): results = predDf
            else: results = results.join(predDf, how='outer')
        else:
            pred = np.squeeze(pred[prediction_index])
            predDf  = pa.Series({chars:pred}, name="Flag {}".format(flag)).to_frame()
            results = results.append(predDf)

    if flag is not None:
        s = results.astype(float)
        if sorting: s = s.sort_values(ascending=True)
        s.plot(kind='barh', color='Blue')
        plt.gca().invert_yaxis()
    else:
        MyPlot.stacked_bar(results)

    plt.title("Results for model {}".format(model.name))
    plt.show()
    return results


def predict_positions(model_info, word, flag=1, max_len=10):
    model, x_dims, normalise = model_info
    raw_obs = list(word)
    pred = predict(model_info, raw_obs, flag=flag, max_len=max_len)
    pred = np.squeeze(pred)
    pVowel = pa.Series(pred, name="P(vowel)")
    xs = pa.Series((raw_obs + list("__________"))[:max_len], name="Letters")
    fig = pVowel.plot(kind='bar', color='Blue')
    fig.set_xticklabels(xs)
    plt.title("Results for model {}".format(model.name))
    plt.show()


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


def print_obs(Xi, y_cats, p_cats, prob_i=None, loss_i=None, acc_i=None, n_time=None, coder=None):
    if (n_time is None):
        n_time = y_cats.shape[0]

    for j in range(n_time):
        Xij = np.squeeze(Xi[j])
        yij = np.squeeze(y_cats[j])
        pred_ij = np.squeeze(p_cats[j])

        # Will print all probs if no coder provided
        prob_str = prob_i[j]

        if coder is not None:
            Xij = coder.inverse_transform(Xij)
            yij = coder.inverse_transform(yij) if yij >= 0 else '.'
            pred_ij = coder.inverse_transform(pred_ij)
            if (prob_i is not None):
                probSr = pa.Series(data=prob_i[j], index=coder.classes_)
                probSr = probSr.nlargest(5)
                prob_str = ''
                for c, v in probSr.iteritems(): prob_str += "'{}' = {:.04f}, ".format(c, v)

        print("'{}' => '{}' ('{}')".format(Xij, pred_ij, yij))

        if (prob_i is not None):
            print("    - Prob = {}".format(prob_str))

        if (np.ndim(loss_i) > 0):
            # Stateful evaluates each time stamp at a time, reporting loss for each
            print("    - Loss = {:.04f}, Accuracy = {:.04f}".format(loss_i[j], acc_i[j]))

        print()

    # Take average over all timesteps
    if (loss_i is not None):
        loss_i = np.mean(loss_i)
        acc_i = np.mean(acc_i)
        print("* Overall Loss = {:.04f}, Accuracy = {:.04f}".format(loss_i, acc_i))

    print()
    print()


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


def pred_counts(model, X, y, n_top=3, results='s', n_obs=None, n_find=None, mask_zero=True, stateful=False, coder=None):
    '''
    :param model:
    :param X:
    :param y:
    :param n_top:
    :param results: Set to 's' for Summary, 'c' for Counts, 'd' for Detail, or combinations e.g. 'scd'
    :param n_obs:
    :param n_find:
    :param mask_zero:
    :param stateful:
    :param coder:
    :return:
    '''
    # model=model5
    details = True if 'd' in results else False
    summary = True if 's' in results else False
    counts  = True if 'c' in results else False

    if n_obs is None: n_obs = X.shape[0]
    n_time = X.shape[1]
    n_cats  = y.shape[2]
    if n_find is None: n_find = n_time
    # Counts summarise for each (prefix size, prediction rank, find position)
    countsD = collections.defaultdict(lambda:0)
    detail = []
    #Xi = X[0,...]
    #yi = y[0,...]
    def update_for_obs(Xi, yi, prob_i, n_time_i):
        #j = 0
        # Cut off y when padding starts (1 before the end, as last prediction is of a padded value)
        y_cats = np.argmax(yi[0:n_time_i-1], axis=1)
        if coder: y_cats = coder.inverse_transform(y_cats-1)
        for j in range(n_time_i):
            # Get predicted categories in descending order
            if coder:
                probSr = pa.Series(data=prob_i[j,1:], index=coder.classes_).sort_values(ascending=False)
            else:
                probSr = pa.Series(data=prob_i[j]).sort_values(ascending=False)

            # Get first part of X upto the prediction
            if coder:
                # Subtract 1 to allow for padding
                x_pfx = coder.inverse_transform(Xi[0:j + 1] - 1)
            else:
                x_pfx = Xi[0:j+1]
           # t=0
            for t in range(n_top):
                next_top_pred = probSr.index[t]
                next_top_prob = probSr.values[t]
                y_rest        = y_cats[j:]
                find_preds    = list(np.where(y_rest==next_top_pred)[0])
                # Check if it has been found, and how far ahead
                if (len(find_preds)>0):
                    f = find_preds[0]
                    # Checi if index where prediction is found is within range
                    if (f < n_find) : countsD[(j,t,f)] += 1
                else:
                    f = -1  # Not found
                if details:
                    detail.append({'Pfx':x_pfx, 'Pred':next_top_pred, 'Prob':next_top_prob, 'n_Pfx':j, 'n_Top':t, 'n_Find':f})

    # i =0
    for i in range(n_obs):
        n_time_i = np.count_nonzero(X[i]) if mask_zero else n_time
        prob_i   = predict_obs(model, X, i, n_time, n_cats, stateful=stateful, mask_zero=mask_zero)
        update_for_obs(X[i,...], y[i,...], prob_i, n_time_i)

    countsDf = pa.Series(countsD).reset_index()
    countsDf.columns = (['n_Pfx','n_Top', 'n_Find', 'Count'])

    results = []


    if summary:
        summaryDf = pa.pivot_table(countsDf, index='n_Top', columns='n_Find', values='Count', aggfunc=np.sum) / np.sum(countsDf['Count'])
        results.append(summaryDf)

    if counts: results.append(countsDf)

    if details:
        detailDf = pa.DataFrame.from_dict(detail)[['Pfx', 'Pred', 'Prob', 'n_Pfx', 'n_Top', 'n_Find']]
        results.append(detailDf)

    return results


def model_evaluate(model, X, y, stateful = False, mask_zero=True, printing=False, coder=None, n_obs=None):
    #model = model5

    float_formatter = lambda x: "%.4f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})

    if n_obs is not None:
        X=X[0:n_obs,...]
        y=y[0:n_obs,...]

    print("Evaluating model '{}'".format(model.name))
    print("X shape: {}".format(X.shape)) # (450, 10, 1)
    print("y shape: {}".format(y.shape)) # (450, 10, 28)
    print()
    n_obs   = X.shape[0]
    n_time  = X.shape[1]
    n_cats  = y.shape[2]
    #i,j = (0,0)
    #i,j = (1,0)

    acc_s = []
    loss_s = []
    i=0
    for i in range(n_obs):
        prob_i         = predict_obs(model, X, i, n_time, n_cats, stateful=stateful, mask_zero=mask_zero)
        loss_i, acc_i  = evaluate_obs(model, X, y, i, n_time, n_cats, stateful=stateful, mask_zero=mask_zero)
        acc_s  += acc_i
        loss_s += loss_i

        if printing:
            # Drop 1st column which is just the prediction of a padded value
            prob_i = prob_i[:, 1:]
            p_cats = np.argmax(prob_i, axis=1)
            # Subtract 1 as predictions include padding value
            X_     = np.squeeze(X - 1)
            y_cats = np.argmax(y[i], axis=1) - 1
            # Print
            print_obs(Xi = X_[i], y_cats=y_cats, p_cats=p_cats, prob_i=prob_i, loss_i=loss_i, acc_i=acc_i, n_time=n_time_i, coder=coder)



    # Average over all observations
    accuracy = (np.mean(acc_s))
    loss     = (np.mean(loss_s))

    print("Model '{}' : loss {:.04f}, accuracy {:0.4f}".format(model.name, loss,accuracy))
    np.set_printoptions(formatter=None)

    #return loss_s, acc_s
    return loss, accuracy








############## Build Models  ###################
max_len = 10
h = []


# First simple model
# - need to train for 100+ epochs, even though accuracy doesn't change much
model_name="vowel1a2"
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len)
(X_train,  y_train), (X_test,y_test) = MyNn.splitData(X, y, [80,20])
model1 = create_model_A(hidden_units=30, input_shape = (max_len, 1), model_name=model_name)
h += model_fit(model1, X, y, epochs=2, batch_size=1)
plt.plot(h)


# More units, more input
model_name="vowel1b"
raw_data, coder = get_raw_data(n_chars=6000)
X,y = get_xy_data(raw_data, coder, max_len=max_len)
model1b = create_model_A(hidden_units=50, input_shape = (max_len, 1))
h += model_fit(model1b, X, y, epochs=20, batch_size=1, model_name=model_name)




# Doesn't work so well with 2 layers
# - Perhaps due to prediction from dummy timestates being treated as genuine
model_name="vowel2"
raw_data, coder = get_raw_data(n_chars=6000)
X,y = get_xy_data(raw_data, coder, max_len=max_len)
model2 = create_model_B(hidden_units=[30,20], input_shape = (max_len, 1), model_name=model_name)
h += model_fit(model2, X, y, epochs=100, batch_size=1)




# Use embedding
# - works well for first few places, then predicts 0
model_name="vowel3"
raw_data, coder = get_raw_data(n_chars=6000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2)
model3 = create_model_C(hidden_units=30, max_len=max_len, num_categories=num_cats, embedding_size=7, model_name=model_name)
h += model_fit(model3, X, y, epochs=100, batch_size=1, model_name=model_name)




# Use embedding and predict vowel & space
# - works well
model_name="4.1_vowel_space"
h = []
raw_data, coder = get_raw_data(n_chars=6000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=2)
model4 = create_model_D(hidden_units=50, max_len=max_len, num_categories=num_cats, embedding_size=20, num_flags=2, model_name=model_name)
h += model_fit(model4, X, y, epochs=2, batch_size=5)
plt.plot(h)



# Use embedding and predict vowel & space
# - Works well
h=[]
model_name="4b_vowel_space"
raw_data, coder = get_raw_data(n_chars=20000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=2)
num_flags = y.shape[2] - 1
model4b = create_model_D(hidden_units=100, max_len=max_len, num_categories=num_cats, embedding_size=30, num_flags=num_flags, model_name=model_name)
h += model_fit(model4b, X, y, epochs=1000, batch_size=3)
model4b.evaluate(X, y, batch_size=50)
plt.plot(h)





# Use embedding to predict next letter
# - Model 5_ used only 3000 obs, 5.1_ used 20000
# - model 5_ shows interesting pattern with k:
#       - k at the start followed by n
#       - ok is v.different

h=[]
model_name="5_2_next_letter"
raw_data, coder = get_raw_data(n_chars=20000)
raw_data, coder = get_raw_data(n_chars=200000, filename="training words.txt")

num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
num_flags = y.shape[2] - 1
model5 = create_model_D(hidden_units=100, max_len=max_len, num_categories=num_cats, embedding_size=50, num_flags=num_flags, model_name=model_name, mask_zero=True)
h += model_fit(model5, X, y, epochs=2, batch_size=5)
model5.evaluate(X, y, batch_size=5)
model_evaluate(model5, X, y, stateful=False, printing=False, mask_zero=True)
model_evaluate(model5, X, y, stateful=False, printing=True, mask_zero=True, n_obs=3)

plt.plot(h)





# Stateful
h=[]
model_name="6_Stateful_next_letter"
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=3, flags=0)
num_flags = y.shape[2] - 1
model6 = create_model_E(hidden_units=100, num_flags=num_flags, model_name=model_name)
#tr_loss, tr_acc = model_fit_stateful(model6, X, y, epochs=1000)
h += model_fit_stateful(model6, X, y, epochs=1000)
plt.plot(h)


model_evaluate(model6, X, y, stateful=True, printing=False, mask_zero=True)
model_evaluate(model6, X, y, stateful=True, printing=True, coder=coder, n_obs=3)
model_evaluate(model6, X, y, stateful=True, printing=False, coder=coder)

# Overstates accuracy when include the padded values
model_evaluate(model6, X, y, stateful=True, printing=False, mask_zero=False)





# Stateful with Embedding
# model 7   - based on   3000 obs and  50 epochs, 10 embeddings
# model 7.1 - based on  20000 obs and 250 epochs, 10 embeddings
# model 7.2 - based on 200000 obs and 250 epochs, 10 embeddings
# Much slower using stateful: 1 epoch with 200k obs takes 5 mins, compared to 30secs for model5
h=[]
model_name="7_2_Stateful_emb_next_letter"
raw_data, coder = get_raw_data(n_chars=20000)
raw_data, coder = get_raw_data(n_chars=200000, filename="training words.txt")
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
num_flags = y.shape[2] - 1
model7 = create_model_F(hidden_units=100, num_flags=num_flags, embedding_size=10, model_name=model_name)
h += model_fit_stateful(model7, X, y, epochs=5)
plt.plot(h)

model_evaluate(model7, X, y, stateful=True, printing=False)
model_evaluate(model7, X, y, stateful=True, printing=True, coder=coder, n_obs=3)




################ Predicting & Evaluating ##################

# Import a model
model1a = model_load("Keep\model_vowel1a_final")
model1b = model_load("Keep\model_vowel1b_final")
model3  = model_load("Keep\model_vowel3_final")
model4  = model_load("Keep\model_4.1_vowel_space_final")
model4b = model_load("Keep\model_4b_vowel_space_final")
model5  = model_load("Keep\model_5_next_letter_final")
model52  = model_load("Keep\model_5_2_next_letter_final")
model6  = model_load("Keep\model_6_Stateful_next_letter_final")
model7  = model_load("Keep\model_7_Stateful_emb_next_letter_final")
model71  = model_load("Keep\model_7_1_Stateful_emb_next_letter_final")
model72  = model_load("Keep\model_7_2_Stateful_emb_next_letter_v80")


x_dim = 2 # Use 3 if no embeddings in model (e.g. model 6)
max_len=10
raw_data, coder = get_raw_data(n_chars=20000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=x_dim, flags=0)






model52.summary()
model6.summary()
model72.summary()

model_evaluate(model52, X, y, stateful=False)
# Model 'Keep\model_5_2_next_letter_final' : loss 1.4773, accuracy 0.6383

X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=3, flags=0)
model_evaluate(model6, X, y, stateful=True)
# Model 'Keep\model_6_Stateful_next_letter_final' : loss 3.1114, accuracy 0.4097

X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
model_evaluate(model72, X, y, stateful=True)
# Model 'Keep\model_7_2_Stateful_emb_next_letter_v80' : loss 2.7270, accuracy 0.4485




# See whether the Top predictions are Found, given diffent length Prefix of sequence
s, c, d = pred_counts(model5, X, y, n_top=3, n_obs=1, coder=coder, results = 'scd')
s, c = pred_counts(model5, X, y, n_top=4, n_find=4, results='sc')
# 33% of Top predictions were found in the next position, 4.6% at position 1
# 10% of 2nd predictions were found in the next position, 6.1% at position 1
# n_Find         0         1         2         3
# n_Top
# 0       0.332552  0.048922  0.030466  0.022791
# 1       0.104289  0.061050  0.038727  0.022967
# 2       0.068022  0.047340  0.029060  0.022322
# 3       0.048922  0.068549  0.032048  0.021971




# Set model info to Model, x_dims, normalise (2 if using embeddings)
model_info = (model1, 3, False)
model_info = (model1b, 3, False)

model_info = (model3, 2, False)
model_info = (model4, 2, False)
model_info = (model4b, 2, False)
model_info = (model5, 2, False)
model_info = (model6, 3, False)
model_info = (model7, 2, False)




# Note: you need a coder
raw_data, coder = get_raw_data(n_chars=3000)

# predict_next_letter(model_info, 'th')

# Use with model5
# Try: "p r e s e r v e"
# The try from "s" -> i..n..g..l..e
# t..o..o..t..h
# m..o..l..a..r
# t..w..e..n..t..y
model5 = model_load("Finals\model_5_next_letter_final")
model_info = (model5, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True)
myUi.ChartUpdater(plot_Fn = plot_next_letter)


model6 = model_load("Finals\model_6_Stateful_next_letter_final")
model_info = (model6, 3, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax)
myUi.ChartUpdater(plot_Fn = plot_next_letter)

model7  = model_load("Keep\model_7_Stateful_emb_next_letter_final")
model_info = (model7, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax, printing=True)
myUi.ChartUpdater(plot_Fn = plot_next_letter)


# Flag None = category index
# Flag 1 = prob(vowel) following
# Flag 2 = prob(space) following
# N.b. For model5, with all letters, flag 1 is ' ' and 2 is 'a'
flag = 1
flag = 2
flag = None

r=predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", '', flag)
r=predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", 'a', flag)



predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", '', flag=None)
predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", 'a')
predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", 'a', flag)
predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", 'th', 2)
predict_each(model_info, "abcdefghijklmnopqrstuvwxyz", 't')



predict_positions(model_info, "hello", flag)
predict_positions(model_info, "formatter", 2)
predict_positions(model_info, "evory")
predict_positions(model_info, "phoning")
predict_positions(model_info, "thrashing", flag)
predict_positions(model_info, "crashing")
predict_positions(model_info, "shing")
predict_positions(model_info, "bananaman")
predict_positions(model_info, "crashing", flag)
predict_positions(model_info, "england", flag)
predict_positions(model_info, "ingland", flag)
predict_positions(model_info, "ingland", flag)
predict_positions(model_info, "answer", flag)




float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})



def setup():


    prefix='o'
    word='abcdef'
    word='abcdef'
    text='ok'
    word='ok'
    raw_item='k'
    word=['a', 'b', 'c']
    flag=None







