
from __future__ import print_function

import re

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
import collections
from tabulate import tabulate as tab
import random
import seaborn as sns


## Getting warnings from sklearn doing coder.inverse_transform(2)
## DeprecationWarning: The truth value of an empty array is ambiguous.
# Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#data_path = r"C:\Temp\TestPyCharm\Data\Models\002 RNN vowel"
models_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Data\Models\002 RNN vowel"
data_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Data\Sample Data"
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Global coder that is used by default if None is set for a model (e.g. if it has been reloaded)
_coder = None

############# Raw Data #############

def read_file(filename=None, n=10000):
    if filename is None : filename = "english text.txt"
    filepath = data_path + "\\" + filename
    print("Reading filename: {}".format(filepath))
    with tf.gfile.GFile(filepath, "r") as f:
        text = f.read(n).replace("\n", " ")
    print("Read {} characters".format(len(text)))
    return text

def randomise_text(text, data_info=None, p_insert=0.0, max_insert=0, insert_from=alphabet, seed=999):
    '''
        Randomly inserts characters into the raw_data
    '''
    if data_info is not None:
        rand_info = data_info.get('randomise')
        if (rand_info is None):
            print("No randomisation requested.")
            return text
        p_insert, max_insert = (rand_info)

    print("Randomising raw_data with prob of {}, inserting upto {} chars from '{}' with seed {}".format(p_insert, max_insert, insert_from, seed))
    random.seed = seed
    rand_data = []
    for c in text:
        rand_data.append(c)
        if random.random() < p_insert:
            for i in range(random.randint(1, max_insert)):
                rand_data.append(random.choice(insert_from))
    print ("Randomised data is now {} chars.".format(len(rand_data)))
    return ''.join(rand_data)

def prep_text(input_text:str, allow=None):
    import string
    if (allow is None): allow = ""

    invalid = string.punctuation + '1234567890' + '-_*—°×–−'
    invalid = ''.join(list(set(invalid) - set(allow)))

    # Each split char separated by |
    splitter = '; |, |\r|\n| '
    x = input_text.lower()
    remove = str.maketrans('', '', invalid)
    x = x.translate(remove)
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

def get_raw_data(data_info=None, n_chars = 1000, filename=None, return_text = False, allow=None):
    if (data_info) :
        print("Using data_info to get_raw_data:\n{}".format(data_info))
        n_chars = data_info['n_chars']
        allow = data_info.get('allow')
        filename = data_info.get('filename')

    text      = read_file(n=n_chars, filename=filename)

    if data_info:
        text  = randomise_text(text, data_info)


    words     = prep_text(text, allow=allow)
    raw_data  = words_to_raw_data(words)



    max_len   = max([len(raw_obs) for raw_obs in raw_data])
    n_obs = len(raw_data)
    print("Got {} words with max length {}".format(n_obs, max_len))

    coder = create_encoder(raw_data)


    if data_info:
        data_info.update({'num_categories': len(coder.classes_) + 1})
        data_info.update({'coder': coder})

    # Used to return coder
    if return_text:
        return raw_data, text, coder
    else:
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


def createY_flag(raw_data, coder, max_len = 10, n_dims=3, flags=1):
    '''
        Flags = 0, returns categorical : 1-hot encoding of X (for next char)
        Flags = 1, returns binary      : 1 for vowels, 0 otherwise (for next char)
        Flags = 2, returns categorical : 1 for vowels, 2 for space, 0 otherwise (for next char)
    :param raw_data:
    :param coder:
    :param max_len:
    :param n_dims:
    :param flags:
    :return:
    '''
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
        # For single flag, just return as binary
        y = shifted
        y_shape = (n_obs, max_len, 1)[0:n_dims]
    else:
        y = keras.utils.to_categorical(shifted)
        n_cats = np.max(shifted) + 1
        y_shape = (n_obs, max_len, n_cats)

    y = y.reshape(y_shape)
    print(y.shape)

    return y

def createY_dists(raw_data, coder, max_len = 10, max_ahead = 1, out_of_range=0,  n_obs=None, distance=False):

    print("Getting Y categories with look ahead of {}".format(max_ahead))

    if n_obs is None: n_obs=len(raw_data)
    n_time = max_len
    n_cats = len(coder.classes_) + 1

    # Encode the categories, adding 1 to make space for 0 to be the padding
    flagged = encode_raw_data(coder, raw_data[0:n_obs], max_len=max_len) + 1
    padded  = K_prep_seq.pad_sequences(flagged, maxlen = max_len, value=0, padding='post')

    def get_dists1(A):
        '''
            First try - retunrs df that needs to be transposed
        :return:
        '''
        obs_ix = np.arange(0, n_obs)
        def dist(p, d):
            pfx  = A[:, 0:p+1]
            cats = A[:, p+d]     # Distance d away
            dists = pa.DataFrame(np.stack([obs_ix, np.full(n_obs,d), cats]).T)
            dists.columns=["Ix","Dist","Cat"]
            return dists

        df = pa.DataFrame()
        for d in range(p+1, n_time):
            df = df.append(dist(p, d))

        df = df.sort_values(by=['Ix', 'Dist'])
        return df

    def get_dists2(A, start_pos=0):
        '''
            Hopefully more efficient
            - Could also use approach of shifting, then turn into 1-hot, shift again and make "2-hot" and combine shifted arrays
            - This current approach is too simplistic as it can't handle the 2nd occurance of a letter
                E.g. in word "Text", the T's position is recorded as 0, so 2nd T is silent.
        :param A:
        :return:
        '''
        df = None
        p,i=(0,0)
        def dist_i(i):
            # Get the categories from the start_pos position to the end, for observation i
            cats = np.trim_zeros(A[i, start_pos:], 'b')
            # Generate distance values 0, 1, 2, 3... for how far each item is from the start pos
            dists1 = np.arange(0, len(cats))
            # Or just populate a 1 for all upcoming categories, if not worried about how far ahead they are
            # - Not so useful if going to generate distances with different prefixes later
            dists2 = np.full(len(cats), 1)
            # Put distances against categories in dataframe for this obs
            distsDf = pa.DataFrame(data=[dists1], index=[i])
            # Columns have to be string to do concat later
            distsDf.columns = list(map(str, cats))
            # Keep first occurrence only of each category
            distsDf = distsDf.T.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
            distsDf.index.rename('Obs', inplace=True)
            return distsDf.T
        for i in range(n_obs):
            # Get the position of each category for this obs
            dfi = dist_i(i)
            # Combine for all observations
            if (df is None):  df = dfi
            else:             df = pa.concat([df, dfi], join="outer")

        df.columns = list(map(int, df.columns))
        df.sort_index(axis=1, inplace=True)
        return df

    # Encodes each observation giving the position of the first occurrence of each item in the observation
    # - the start_pos is given a value of 0, the character at the next position is given a value of 1 etc
    dists = get_dists2(padded)

    # Add columns for any categories that haven't been seen in the observations processed
    dists = dists.reindex(list(range(0, n_cats)), axis=1)


    def get_y_hot():
        '''
            Gets the required output vector y for position p in the input vector
        '''
        # Initialise df that will be repeatedly processes
        d_iter = dists
        # Create empty array based on the number of categories that have been found in the dists
        n_cats = len(dists.columns)
        y = np.empty((n_obs, n_time, n_cats))
        for p in range(max_len):
            # Blank out distances to the current item p
            d_iter.replace(0, np.NaN, inplace=True)
            # Set this as the distances for position p
            y[:, p, :] = d_iter
            # Subtract 1 to remove next item in the sequence
            d_iter = d_iter.subtract(1)
        return y

    # Get distances to each category in 1-hot style
    y = get_y_hot()

    # Blank out where too far ahead
    if distance:
        print("Using distance to next character")
        # For distance return 1/position (e.g. for distance 1,2,3 predicting 1, 0.5, 0.3333)
        # - NaN otherwise so loss is 0
        transform_y = np.vectorize(lambda y: 1/y if y <= max_ahead else out_of_range)
        y_ = transform_y(y)
    else:
        # For non-distance, always predict 1 for any positions in range, 0 otherwise
        transform_y = np.vectorize(lambda y: 1 if y <= max_ahead else np.NaN)
        y_ = np.nan_to_num(transform_y(y))

    def check(y_check, obs=0):
        r_obs = pa.DataFrame(data = coder.transform(raw_data[obs]) + 1, index = raw_data[obs]).T
        d_obs = dists.iloc[obs:obs+1,:].dropna(axis=1)
        y_obs = pa.DataFrame(data= y_check[obs, ...])
        y_obs = y_obs.dropna(axis=1, how='all')
        print(tab(r_obs, headers='keys'))
        print()
        print(tab(d_obs, headers='keys'))
        print()
        print(tab(y_obs, headers='keys'))
    # check(y, 1)
    # check(y_, 1)

    return y_

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

def get_xy_data(raw_data, coder=None, data_info=None, max_len=10, normalise=True, x_dims=3, y_dims=3, flags=1, max_ahead=None):
    '''
        Flags = None, returns categorical : 1-hot encoding of X (for next ' max_ahead' chars)
            - Uses max_ahead to get 1's for each of the next chars
        Flags = 0, returns categorical : 1-hot encoding of X (for next char)
        Flags = 1, returns binary      : 1 for vowels, 0 otherwise (for next char)
        Flags = 2, returns categorical : 1 for vowels, 2 for space, 0 otherwise (for next char)
    :param raw_data:
    :param coder:
    :param max_len:
    :param normalise:
    :param x_dims:
    :param y_dims:
    :param flags:
    :param max_ahead:
    :return:
    '''
    if (data_info):
        max_len     = data_info.get('max_len')
        normalise   = data_info.get('normalise')
        x_dims      = data_info.get('x_dims')
        y_dims      = data_info.get('y_dims')
        flags       = data_info.get('flags')
        max_ahead   = data_info.get('max_ahead')
        distance    = data_info.get('distance')
        out_of_range= data_info.get('out_of_range')
        coder       = data_info.get('coder')


    max_len   = min(max_len, max([len(raw_obs) for raw_obs in raw_data]))
    n_obs = len(raw_data)
    print("Applying max length {} to {} words".format(max_len, n_obs))
    print(raw_data[:4])

    X = createX(raw_data, coder, max_len = max_len, normalise=normalise, n_dims=x_dims)
    if (max_ahead is not None) & (flags is None):
        y = createY_dists(raw_data, coder, max_len = max_len, max_ahead=max_ahead, out_of_range=out_of_range, distance=distance)
    else:
        # Can do 1-ahead if flags is None
        y = createY_flag(raw_data, coder, max_len = max_len, n_dims=y_dims, flags=flags)

    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))

    if data_info:
        data_info.update({'num_flags': y.shape[2] - 1})

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


def create_model_D(hidden_units, max_len, num_categories, embedding_size, num_flags, dropout=None, mask_zero=False, model_name="ModelD"):
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
    model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout))
    model.add(TimeDistributed(Dense(num_flags+1, activation = "softmax")))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


def create_model_D_(model_info):
    '''
        Input has to be 2 dimensions: n_obs * n_time_stamp (with no n_features)
        Output is categorical
    :param hidden_units:
    :param max_len:
    :param num_categories:
    :param embedding_size:
    :return:
    '''

    mi = model_info
    di = model_info['data_info']
    model = Sequential(name=mi['name'])
    model.add(Embedding(input_dim=di['num_categories'], input_length=di['max_len'], output_dim=mi['embedding_size'], mask_zero=mi['mask_zero'])) # , dropout=0.2, mask_zero=True))
    model.add(LSTM(mi['hidden_units'], return_sequences=True, dropout=mi['dropout']))
    model.add(TimeDistributed(Dense(di['num_flags']+1, activation = "softmax")))
    model.compile(loss=model_info['loss'], optimizer='adam', metrics=['accuracy'])

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

def model_load(model_name_or_info):
    if (type(model_name_or_info) is dict):
        model_name = "\\keep\\model_{}_final".format(model_name_or_info['name'])
    else:
        model_name = model_name_or_info

    filepath = "{}\\{}.hdf5".format(models_path, model_name)
    print("Loading model from {}".format(filepath))
    model = load_model(filepath)
    # Name doesn't get saved
    model.name = model_name
    print("Loaded model {}".format(model.name))

    if (type(model_name_or_info) is dict):
        model_name_or_info['model'] = model

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



_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def loss_2_stage(y_true, y_pred):
    y_pred = np.clip(y_pred, _Epsilon, 1-_Epsilon)
    if (np.isnan(y_true)):
        # Doesn't matter what prediction is, if there is no position to predict
        loss = 0
    else:
        loss = (y_true - y_pred) ** 2
    return loss

def loss_2_stage_np_(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1-_EPSILON)
    return np.where(np.isnan(y_true), 0, (y_true - y_pred) ** 2)

def loss_2_stage_K__(y_pred, y_true):
    '''
        Loss for a vector
        I actually tried that a few hours after posting. And I was puzzled by what I saw.
        Keras did not give me an error, but the loss went immediately to NaN.
        Eventually I solved the problem. The calling convention for a Keras loss function is first y_true, then y_pred -- or as I call them, tgt and pred.
        But the calling convention for a TensorFlow loss function is pred first, then tgt
    '''
    y_pred = K.clip(y_pred, _EPSILON, 1-_EPSILON)
    no_loss = tf.zeros_like(y_pred)
    mse_loss = tf.square(tf.subtract(y_true, y_pred))
    l1_loss = tf.subtract(y_true, y_pred)
    #return tf.where(tf.is_nan(y_true), no_loss, mse_loss) #, no_loss, mse_loss
    return tf.where(tf.is_nan(y_true), no_loss, mse_loss) #, no_loss, mse_loss

    #return tf.where(tf.is_nan(y_true), 0, (y_true - y_pred) ** 2)

def loss_2_stage_K_(y_true, y_pred):
    '''
        Loss for a vector
    '''
    y_pred = K.clip(y_pred, _EPSILON, 1-_EPSILON)
    loss  = tf.abs(tf.subtract(y_true, y_pred))
    loss2  = tf.square(loss)
    loss0 = tf.multiply(y_true, loss2)
    return loss0

def loss_2_stage_K(y_true, y_pred):
    '''
        Mean Loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    print("Using loss 2 stage")
    loss = K.mean(loss_2_stage_K_(y_true, y_pred))
    #loss2 = tf.where(tf.is_nan(loss), tf.constant(99.0), tf.constant(11.0))
    return loss

def test_loss_np():
    y_true = np.array([np.NaN, np.NaN, 0.0,  0.0,  1.0,  1.0])
    y_pred = np.array([0.0   , 1.0   , 0.1,  0.9,  0.1,  0.9])
    expect = np.array([0.0   , 0.0   , 0.01, 0.81, 0.81, 0.01])
    loss_2_stage_np_(y_true, y_pred) == expect

def test_loss_K():
    #loss_fn = keras.losses.categorical_crossentropy
    loss_fn = loss_2_stage_K_


    y_true = K.constant(np.array([np.NaN, np.NaN, 0.0,  0.0,  1.0,  1.0]))
    #y_true = K.constant(np.array([0,0, 0.0,  0.0,  1.0,  1.0]))
    y_pred = K.constant(np.array([0.0   , 1.0   , 0.1,  0.9,  0.1,  0.9]))
    loss_tf = loss_fn(y_true, y_pred)
    expect = np.array([0.0   , 0.0   , 0.01, 0.81, 0.81, 0.01])
    tfs = tf.Session()
    actual = tfs.run(loss_tf)
    #actual,a,b = tfs.run(loss_tf)
    diff = np.sum(np.abs(actual - expect))
    diff < 1E-5

def test_loss_np():
    y_true = np.array([np.NaN, np.NaN, 0.0,  0.0,  1.0,  1.0])
    y_pred = np.array([0.0   , 1.0   , 0.1,  0.9,  0.1,  0.9])
    expect = np.array([0.0   , 0.0   , 0.01, 0.81, 0.81, 0.01])
    loss_2_stage_np_(y_true, y_pred) == expect

def test_loss():
    v_small = 1E-5
    v_large = 0.999
    abs(loss_2_stage(y_true = np.NaN, y_pred = 0))   == 0
    abs(loss_2_stage(y_true = np.NaN, y_pred = 1))   == 0

    # Truth is 1 -> MSE error
    abs(loss_2_stage(y_true = 1.0,    y_pred = 1.0)) < v_small
    abs(loss_2_stage(y_true = 1.0,    y_pred = 0.1)) == 0.9**2
    abs(loss_2_stage(y_true = 1.0,    y_pred = 0.0)) > v_large

    # Truth is 0 -> MSE error
    abs(loss_2_stage(y_true = 0.0,    y_pred = 0.0)) < v_small
    abs(loss_2_stage(y_true = 0.0,    y_pred = 0.9)) == 0.9**2
    abs(loss_2_stage(y_true = 0.0,    y_pred = 1.0)) > v_large




def model_fit(model, X, y, epochs, batch_size, model_info=None, stateful=False, shuffle=True, save=True, model_name = "Unknown Model"):
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


    if model_info:
        x_shape = X.shape
        x_shape = [x_shape[0], x_shape[1], 1][0:model_info['data_info']['x_dims']]
        X.reshape(x_shape)
        model_name = model.name

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
                      #,callbacks=[metrics]
                      ).history

        # Got error on callback with dropout i
        #print("{} {:4d} : loss {:.04f}, accuracy {:0.4f}, Precision {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], h['MyPrecision'][0], time.ctime()))
        print("{} {:4d} : loss {:.04f}, accuracy {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], time.ctime()))
        accuracy += h['acc']
        #precision += h['MyPrecision']
        loss += h['loss']

        # When not stateful, state is reset automatically after each input
        # When stateful, this is suppressed, so must manually reset after the epoch (effectively the one big sequence)
        if stateful: model.reset_states()

        if save: model_save(model, models_path, model_name, "latest")

        if not (epoch % 10):
            if save: model_save(model, models_path, model_name, epoch)


    if save: model_save(model, models_path, model_name, "final", echo=True, temp=False)

    if model_info:
        model_info.update({'model':model})

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

def check_x_shape(X, data_info):
    x_dims  = data_info['x_dims']
    max_len = data_info['max_len']
    x_shape = (X.shape[0], max_len, 1)[0:x_dims]
    if (X.shape != x_shape):
        print("Changing X shape from {} to {}".format(X.shape, x_shape))
        X = X.reshape(x_shape)
    return X


def predict(model_info_or_tuple, word, flag=1, max_len=10, printing=False, coder=None):
    '''
         model_info = model41_info
         word="hello"

    '''
    raw_obs = list(word)

    if (type(model_info_or_tuple) is dict):
        model, x_dims, normalise = (model_info_or_tuple['model'], model_info_or_tuple['data_info']['x_dims'], model_info_or_tuple['data_info']['normalise'])
        coder = model_info_or_tuple['data_info'].get('coder')
        max_len = model_info_or_tuple['data_info']['max_len']
    else:
        model, x_dims, normalise = model_info_or_tuple

    if (coder is None):
        print("********* No coder has been generated yet for this model.  Using global coder _coder *********")
        coder = _coder

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


def predict_next_letter(model_info_or_tuple, text, ax=None, top_n=None, printing=False, cutoff=100, coder=None):
    '''
    :param model_info_or_tuple:
    :param letters:
    :param prefix:
    :param max_len:
    :param flag: Which category to give the probability for.  If None then returns the argmax category
    :return:
    model_info = model8_5c_info
    text="<"
    text="t"
    printing=False
    '''

    if ax is None:
        ax = plt.gca()
    else:
        ax.clear()

    if (type(model_info_or_tuple) is dict):
        max_len = model_info_or_tuple['data_info']['max_len']
        distance = model_info_or_tuple['data_info'].get('distance')
    else:
        max_len  = 10
        distance = False

    raw_obs = list(text)
    prediction_index = len(text)-1
    pred = predict(model_info_or_tuple, raw_obs, flag=None, max_len=max_len, printing=printing, coder=coder)

    if (distance):
        pred = 1 / pred
        pred = pred.applymap(lambda x: x if x<=cutoff else np.NaN)
        # Converts to np - no longer a df
        #pred = np.where(pred<=10, pred, 0)

    # Keep 5 most likely next letters
    predCol = pred.columns[prediction_index]
    if top_n:
        predDf = pred.nlargest(top_n, predCol)[[predCol]]
    else:
        predDf = pred[[predCol]]
    predDf.columns = [text]


    results = predDf
    if distance:
        sns.stripplot(y=results[text], x=results.index.values, hue=results[text], ax=ax)
        ax.legend().set_visible(False)
        ax.grid(axis='x')
    else:
        ax.bar(results.index.values, results[text])

    ax.set_title ("Probability of letter following: {}...".format(text))
    #ax.set_title ("Results for model {}".format(model.name))


def predict_next_letter_stateful(model_info, text, ax=None, top_n=None, printing=False, coder=None):
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
        pred = predict(model_info, [letter], flag=None, max_len=1, printing=printing, coder=coder)


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


def predict_positions(model_info, word, flag=1, max_len=10, coder=None):
    # model_info=model41_info
    # word="hello"
    # , flag, coder
    model, x_dims, normalise = model_info
    raw_obs = list(word)
    pred = predict(model_info, raw_obs, flag=flag, max_len=max_len, coder=coder)
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


def pred_counts(model_or_info, X, y, n_top=3, results='s', n_obs=None, n_find=None, mask_zero=True, stateful=False, coder=None):
    '''
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
    :return:
    '''

    if (type(model_or_info) is dict):
        model = model_or_info['model']
        check_x_shape(X, model_or_info['data_info'])
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
    summaryDf = pa.pivot_table(countsDf, index='n_Top', columns='n_Find', values='Count', aggfunc=np.sum) / np.sum(countsDf['Count'])

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












############## Build Models - individual specification, without model_info ###################

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
h += model_fit(model1b, X, y, epochs=2, batch_size=1, model_name=model_name)




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
model_name, dropout =("5_3a_next_letter_NoDropout", 0.0)
model_name, dropout =("5_3b_next_letter_40Dropout", 40.0)
model_name, dropout =("5_2b_next_letter_40Dropout", 40.0) # 200000
raw_data, coder = get_raw_data(n_chars=20000)
raw_data, coder = get_raw_data(n_chars=200000, filename="training words.txt")

num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
num_flags = y.shape[2] - 1
model5 = create_model_D(hidden_units=100, max_len=max_len, num_categories=num_cats, embedding_size=50, num_flags=num_flags, model_name=model_name, mask_zero=True, dropout=dropout)
h += model_fit(model5, X, y, epochs=200, batch_size=5)
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









####################   Using ModelInfo to specify models   ##################################

def data_with(data_settings):
    data_info = {'n_chars': None, 'max_len': 10,
                    'x_dims': 2, 'y_dims': 2,
                    'flags': None, 'max_ahead': 1, 'distance':False,
                    'normalise': False, 'randomise': None
                    }
    data_info.update(data_settings)
    return data_info

def model_with(model_settings):
    model_info = {'create_fn': create_model_D_, 'name':"default_model",
                          'dropout':0.0, 'hidden_units':100, 'embedding_size':50,
                          'loss':keras.losses.categorical_crossentropy,
                          'mask_zero':True}

    model_info.update(model_settings)
    return model_info

def load_data(model_or_data_info):
    if (model_or_data_info.get('data_info') is None):
        data_info = model_or_data_info
    else:
        data_info = model_or_data_info['data_info']

    raw_data, text, coder = get_raw_data(data_info, return_text=True)
    X, y = get_xy_data(raw_data, coder=coder, data_info=data_info)

    data_info['coder'] = coder

    return X,y, text, coder

def build_model(model_info, data_only = False, extract=True, create=False, fit=True, epochs=100):
    if extract | data_only:
        X, y, text, coder = load_data(model_info)
        if data_only:
            return X, y, text

    if create:
        print("Creating new model {}".format(model_info['name']))
        model = model_info['create_fn'](model_info)
    else:
        print ("Reusing existing model {}".format(model_info['name']))
        model = model_info['model']

    if fit:
        model_fit(model, X, y, model_info=model_info, epochs=epochs, batch_size=5)
        model.evaluate(X, y, batch_size=5)

    #return model, X, y



# Define models : Looking ahead & Random noise ##########################

data_info0  = data_with({'n_chars':3000,  'max_len':10,  'flags':0,    })
data_info1  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':1 })
data_info1b = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':1 })
data_info1c = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':1, 'randomise':(0.5, 1)})
data_info2  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 })
data_info2a = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':2 })
data_info2b = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':3, 'randomise':(0.5, 1)})
data_info2c = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2, 'randomise':(0.5, 1)})
data_info2d = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2, 'randomise':(0.5, 1)})
data_info2e = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2 })
data_info3  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':3 })
data_info4  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 })
data_info5  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'distance':True })
data_info5a = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5b = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5c = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5d = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5e = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})


model8_0_info  = model_with({'name':"8_0_next_char_old",        'data_info':data_info0 })
model8_1_info  = model_with({'name':"8_1_next_char_1",          'data_info':data_info1,    'loss':loss_2_stage_K})
model8_1b_info = model_with({'name':"8_1b_next_char_1",         'data_info':data_info1b})
model8_1c_info = model_with({'name':"8_1c_next_char_1_rnd",     'data_info':data_info1c})
model8_2_info  = model_with({'name':"8_2_next_char_2",          'data_info':data_info2 })
model8_2a_info = model_with({'name':"8_2a_next_char_2_20k",     'data_info':data_info2a})
model8_2b_info = model_with({'name':"8_2a_next_char_3_rand20k", 'data_info':data_info2b})
model8_2c_info = model_with({'name':"8_2a_next_char_2_rand20k", 'data_info':data_info2c})
model8_2d_info = model_with({'name':"8_2d_next_char_1_rnd_40do",'data_info':data_info2d, 'dropout':0.4})
model8_2d6_info= model_with({'name':"8_2d_next_char_1_rnd_60do",'data_info':data_info2d, 'dropout':0.6})
model8_2e_info = model_with({'name':"clean data",               'data_info':data_info2e, 'dropout':0.4})
model8_3_info  = model_with({'name':"8_3_next_char_3",          'data_info':data_info3 })
model8_4_info  = model_with({'name':"8_4_next_char_4",          'data_info':data_info4 })

model8_5_info  = model_with({'name':"8_5_distance_to",          'data_info':data_info5 , 'dropout':0.4, 'loss':keras.losses.mse})
model8_5a_info = model_with({'name':"8_5a_distance_to_4",       'data_info':data_info5a, 'dropout':0.4, 'loss':keras.losses.mse})
# Too simplistic to just model distance as 1/d
# E.g. if predicting what follows 't' if will be averaging cases where 'e' is next (y=1), or position 2 (y=0.5) or not at all (y=0.0)
# - in cases where X doesn't occur, the penalty is v.high for predicting say 1, when the truth is 0
# - e.g. for "<Text>" the T is never predicted as 90% of the time it never occurs
# - Better to have a 2 stage prediction:
#   - Predict 1/0 for if X occurs at all (with a binary-cross-entropy loss)
#   - In cases where thruth is 0, no further loss
#   - In cases where thruth is 1, mse loss based on prediction of where.
model8_5b_info = model_with({'name':"8_5b_2stage_distance_to_4",        'data_info':data_info5b, 'dropout':0.4, 'loss':loss_2_stage_K})
model8_5c_info = model_with({'name':"8_5c_2stage_distance_to_4",        'data_info':data_info5c, 'dropout':0.4, 'loss':loss_2_stage_K})
model8_5d_info = model_with({'name':"8_5d_2stage_distance_to_4",        'data_info':data_info5d, 'dropout':0.4, 'loss':loss_2_stage_K})
model8_5e_info = model_with({'name':"8_5e_2stage_distance_to_4",        'data_info':data_info5e, 'dropout':0.4, 'loss':loss_2_stage_K})



#### Run build process for each model #####

build_model(model8_0_info, create=True)
build_model(model8_1_info, create=True, epochs = 10)
build_model(model8_1b_info, create=True)     # Look ahead of 1 - no noise
build_model(model8_1c_info, create=True)     # Look ahead of 1 - with noise
build_model(model8_2_info, create=True)
build_model(model8_2a_info, create=True)
build_model(model8_2b_info, create=True)
build_model(model8_2c_info, create=True)     # Look ahead of 2 - with noise of 1
build_model(model8_2d_info, create=True)     # Look ahead of 2 - with noise & dout
build_model(model8_2d6_info, create=True)     # Look ahead of 2 - with noise & dout
build_model(model8_3_info, create=True)
build_model(model8_4_info, create=True)
build_model(model8_5_info, create=False, fit=True, epochs=100)

build_model(model8_5a_info, create=True)
build_model(model8_5a_info, create=False, epochs=25)
build_model(model8_5b_info, create=True, epochs=200)
build_model(model8_5c_info, create=False, epochs=100)
build_model(model8_5d_info, create=True, epochs=200)
build_model(model8_5e_info, create=True, epochs=200)






##### Reload previous models
model_load(model8_1_info)
model_load(model8_2_info)
model_load(model8_2a_info)
model_load(model8_2c_info)
model_load(model8_4_info)
model_load(model8_5_info)
model_load(model8_5a_info)
model_load(model8_5c_info)







##### View charts for saved models  ####
# Run the model_info above to define model

def prediction_chart(model_info):
    model_load(model_info) # Set 'model'
    load_data(model_info)  # Set 'coder'
    plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True)
    myUi.ChartUpdater(plot_Fn = plot_next_letter)



# Reload previous models
prediction_chart(model8_1_info)
prediction_chart(model8_2_info)
prediction_chart(model8_2a_info)
prediction_chart(model8_2c_info)
prediction_chart(model8_4_info)
prediction_chart(model8_5_info)
prediction_chart(model8_5a_info)
prediction_chart(model8_5c_info)
# => Much more mixed predictions: e.g. after r is not just vowels
# => Much more mixed predictions: e.g. q is followed by a,e,i as well as u


# When using randomised data:
# - becomes hard to see real patterns through the noise


# model8_4_info works well with max_ahead=4, but not using distances, just 1/0

# Can't get actual distance values to work well
# - Perhaps need to predict tuple (prob, dist) and have a better 2-stage loss function



# Use with model5
# Try: "p r e s e r v e"
# The try from "s" -> i..n..g..l..e
# t..o..o..t..h
# m..o..l..a..r
# t..w..e..n..t..y
raw_data, coder = get_raw_data(n_chars=3000)
model5 = model_load("Keep\model_5_2b_next_letter_40Dropout_v165")
model5 = model_load("Keep\model_5_next_letter_final")
model_info = (model5, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)










########### Create summary counts for each model #############

def model_summary(model_info):
    model_load(model_info) # Set 'model'
    # Load clean data - will need different format of X with 2 or 3 dims depending on model
    X, y, text, _coder = load_data(model_info)
    pred_counts(model_info,  X, y, n_top=4, n_find=4, results='s')


model_summary(model8_1_info)
model_summary(model8_2e_info)
model_summary(model8_4_info)
model_summary(model8_5_info)
model_summary(model8_5a_info)
model_summary(model8_5b_info)
model_summary(model8_5d_info)
model_summary(model8_0_info)
model_summary(model8_1_info)
model_summary(model8_1b_info)
model_summary(model8_1c_info)
model_summary(model8_2_info)
model_summary(model8_2a_info)
model_summary(model8_2b_info)
model_summary(model8_2c_info)
model_summary(model8_2d_info)
model_summary(model8_2d6_info)
model_summary(model8_3_info)
model_summary(model8_4_info)   # Works quite well


pred = predict(model8_5_info, 't', flag=None, max_len=10, printing=True)


# Print summaries saved above
print(model8_1_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.456943  0.034954  0.024786  0.015253
# 1       0.069908  0.089291  0.035272  0.021290
# => High prediction 45% for position 0, but poor after that

print(model8_2_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.265836  0.172378  0.023884  0.010644
# 1       0.134735  0.106698  0.039720  0.014798
# => Good prediction 26%, 16% for positions 0 and 1, but poor after that

print(model8_3_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.221060  0.138816  0.069646  0.011172
# 1       0.115046  0.092465  0.066793  0.019729
# => Good prediction 26%, 16% for positions 0 and 1, and better 6.9% for 2




# No noise - look ahead of 1
# => Really focusses on next value and gets it well
print(model8_1b_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.490836  0.026707  0.017543  0.013877
# 1       0.080017  0.073000  0.032468  0.022884
# 2       0.037390  0.046292  0.027702  0.019271


# With noise - look ahead of 1
# - still manages to make a decent prediction overall
# Low frequency relationships (e.g. q...) are lost
# But common ones like "th..e" still picked up.
print(model8_1c_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.447218  0.066692  0.020492  0.013827
# 1       0.056922  0.079690  0.031545  0.023804
# 2       0.034112  0.060440  0.027736  0.019622

# Prediction below of clean data
# n_Find         0         1         2         3
# n_Top
# 0       0.313852  0.058677  0.047393  0.026926
# 1       0.120778  0.053696  0.036576  0.024280
# 2       0.072763  0.045837  0.030661  0.022879



# Model - look ahead of 2
# => Attention spread over 2 values, so not so good at pos=0
print(model8_2a_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.299817  0.158680  0.018591  0.008641
# 1       0.115737  0.120712  0.035350  0.018329
# 2       0.031684  0.030636  0.046347  0.020162


# Look ahead of 2 with random noise
# - Still good at finding position 0, despite noise
# - with noise, often the related item will now be at position 1
print(model8_2c_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.271573  0.070701  0.045065  0.032318
# 1       0.113728  0.054021  0.038904  0.026300
# 2       0.088708  0.045918  0.036203  0.025494


print(model8_2d_info['summary'])
# Tested on clean data. having built with drop out
# - is slightly better than no dropout
# - chart of next is much cleaner
# n_Find         0         1         2         3
# n_Top
# 0       0.310582  0.037420  0.028083  0.020483
# 1       0.156702  0.047119  0.027432  0.021714
# 2       0.124855  0.036769  0.031630  0.016358


# Not much different with 60% dropout - worse if anything
print(model8_2d6_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.309816  0.040591  0.031479  0.022228
# 1       0.148695  0.042455  0.030581  0.019605
# 2       0.116595  0.049772  0.026715  0.019122
















#############  Interactively see most likely next letters - earlier models #####################


# Set model info to Model, x_dims, normalise (2 if using embeddings)
model_info = (model1, 3, False)
model_info = (model1b, 3, False)
model_info = (model3, 2, False)
model_info = (model4, 2, False)
model_info = (model4b, 2, False)
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
model5 = model_load("Keep\model_5_next_letter_final")
model_info = (model5, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)

# With model5.2
# Try: p..r..e..s..i..d..e..n..t
# Try: r : only followed by vowels
# can do r..n..a caused by international
# - xecutions



model6 = model_load("Keep\model_6_Stateful_next_letter_final")
model_info = (model6, 3, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)



model7  = model_load("Keep\model_7_Stateful_emb_next_letter_final")
model_info = (model7, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax, printing=True, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)






# Flag None = category index
# Flag 1 = prob(vowel) following
# Flag 2 = prob(space) following
# N.b. For model5, with all letters, flag 1 is ' ' and 2 is 'a'
flag = 1
flag = 2
flag = None


# Prob of next letter
model5_info = (model5, 2, False)
r=predict_each(model5_info, "abcdefghijklmnopqrstuvwxyz", '', None)


model41 = model_load("Keep\model_4.1_vowel_space_final")
model41_info = (model41, 2, False)
# Probability of vowel next
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", '', 1)
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", 't', 1)
# Prob of space
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", '', 2)
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", 't', 2)
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", 'th', 2)



# Prob of v
model1b = model_load("Keep\model_vowel1b_final")
model1b_info = (model1b, 3, False)
r=predict_each(model1b_info, "abcdefghijklmnopqrstuvwxyz", '', 1)


model1a = model_load("Keep\model_vowel1a_final")
model1a_info = (model1a, 3, False)
r=predict_each(model1a_info, "abcdefghijklmnopqrstuvwxyz", '', 1)
r=predict_each(model1a_info, "abcdefghijklmnopqrstuvwxyz", '', 2)


# Predict prob of vowel (flag=1) after each letter in word
predict_positions(model41_info, "hello", 1, coder=coder)

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







################ Predicting & Evaluating - earlier models ##################

# Best model is model5_2


# Import one model
model1a = model_load("Keep\model_vowel1a_final")
model1b = model_load("Keep\model_vowel1b_final")
model3  = model_load("Keep\model_vowel3_final")
model4  = model_load("Keep\model_4.1_vowel_space_final")
model4b = model_load("Keep\model_4b_vowel_space_final")
model5  = model_load("Keep\model_5_next_letter_final")
model52  = model_load("Keep\model_5_2_next_letter_final")
model52b = model_load("Keep\model_5_2b_next_letter_40Dropout_v165")
model6  = model_load("Keep\model_6_Stateful_next_letter_final")
model7  = model_load("Keep\model_7_Stateful_emb_next_letter_final")
model71  = model_load("Keep\model_7_1_Stateful_emb_next_letter_final")
model72  = model_load("Keep\model_7_2_Stateful_emb_next_letter_v80")


model52.summary()
model6.summary()
model72.summary()



### Load in data for use in evaluation
x_dim = 2 # Use 3 if no embeddings in model (e.g. model 6)
max_len=10
raw_data, coder = get_raw_data(n_chars=20000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=x_dim, flags=0)


model_evaluate(model52, X, y, stateful=False)
# Model 'Keep\model_5_2_next_letter_final' : loss 1.4773, accuracy 0.6383

X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=3, flags=0)
model_evaluate(model6, X, y, stateful=True)
# Model 'Keep\model_6_Stateful_next_letter_final' : loss 3.1114, accuracy 0.4097

X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
model_evaluate(model72, X, y, stateful=True)
# Model 'Keep\model_7_2_Stateful_emb_next_letter_v80' : loss 2.7270, accuracy 0.4485


max_len=10
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
model5 = model_load("Keep\model_5_2b_next_letter_40Dropout_v165")
model5 = model_load("Keep\model_5_next_letter_final")
# See whether the Top predictions are Found, given diffent length Prefix of sequence
s, c, d = pred_counts(model5, X, y, n_top=3, n_obs=1, coder=coder, results = 'scd')
s, c    = pred_counts(model5, X, y, n_top=4, n_find=4, results='sc')
p       = pred_counts(model5, X, y, n_top=4, n_find=4, results='p')

# 33% of Top predictions were found in the next position, 4.6% at position 1
# 10% of 2nd predictions were found in the next position, 6.1% at position 1
# n_Find         0         1         2         3
# n_Top
# 0       0.332552  0.048922  0.030466  0.022791
# 1       0.104289  0.061050  0.038727  0.022967
# 2       0.068022  0.047340  0.029060  0.022322
# 3       0.048922  0.068549  0.032048  0.021971

# Accuracy increases based on the number of preceding characters
pct_by_pfx = p[p['n_Top']==0][['n_Pfx','Pct0']]
plt.bar(pct_by_pfx['n_Pfx'], pct_by_pfx['Pct0'])
plt.show


# The 3000  data was used in training & the No Dropout model does best on this
# The 20000 data was new & the Dropout model does better on this as it has learned more general features

# raw_data, coder = get_raw_data(n_chars=3000)     # raw_data, coder = get_raw_data(n_chars=20000)
# model_5_2b_next_letter_40Dropout_v165            # model_5_2b_next_letter_40Dropout_v165
# n_Find         0         1         2         3   # n_Find         0         1         2         3
# n_Top                                            # n_Top
# 0       0.387183  0.045280  0.026861  0.020721   # 0       0.414216  0.029513  0.024221  0.018111
# 1       0.098235  0.056408  0.034536  0.020721   # 1       0.114233  0.057553  0.035077  0.020184
# 2       0.076746  0.038373  0.033001  0.017268   # 2       0.065463  0.040532  0.031586  0.020839
# 3       0.053722  0.035303  0.026477  0.029163   # 3       0.044242  0.036059  0.027331  0.020839

# model_5_next_letter_final                        # model_5_next_letter_final
# n_Find         0         1         2         3   # n_Find         0         1         2         3
# n_Top                                            # n_Top
# 0       0.510204  0.011303  0.008791  0.010047   # 0       0.332552  0.048922  0.030466  0.022791
# 1       0.064992  0.086028  0.032967  0.016954   # 1       0.104289  0.061050  0.038727  0.022967
# 2       0.034537  0.053689  0.027943  0.011303   # 2       0.068022  0.047340  0.029060  0.022322
# 3       0.021350  0.066248  0.028885  0.014757   # 3       0.048922  0.068549  0.032048  0.021971
























