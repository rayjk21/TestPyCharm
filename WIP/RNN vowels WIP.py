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


## Getting warnings from sklearn doing coder.inverse_transform(2)
## DeprecationWarning: The truth value of an empty array is ambiguous.
# Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


data_path = r"C:\Temp\TestPyCharm\Data\Models\002 RNN vowel"

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


def get_raw_data(n_chars = 1000):
    text      = read_file(n=n_chars)
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model





def model_load(model_name):
    filepath = "{}\\{}.hdf5".format(data_path, model_name)
    print("Loading model from {}".format(filepath))
    model = load_model(filepath)
    # Name doesn't get saved
    model.name = model_name
    print("Loaded model {}".format(model.name))
    return model


def model_save(model, data_path, model_name, sfx, echo=False, temp=True):
    if temp==True:
        temp_path="\\temp"
    else:
        temp_path=""
    filename = "{}{}\\model_{}_{}.hdf5".format(data_path, temp_path, model_name, sfx)
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

        print("{} {:4d} : loss {:.04f}, accuracy {:0.4f}, Precision {:0.4f}".format(lbl, epoch, h['loss'][0], h['acc'][0], h['MyPrecision'][0]))
        accuracy += h['acc']
        precision += h['MyPrecision']
        loss += h['loss']

        # When not stateful, state is reset automatically after each input
        # When stateful, this is suppressed, so must manually reset after the epoch (effectively the one big sequence)
        if stateful: model.reset_states()

        if save: model_save(model, data_path, model_name, "latest")

        if not (epoch % 10):
            if save: model_save(model, data_path, model_name, epoch)


    if save: model_save(model, data_path, model_name, "final", echo=True, temp=False)
    return precision

def model_fit_stateful(model, X, y, epochs, save=True, mask_zero=True):
    #model = model6
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

        print("Epoch {:4d} : loss {:.04f}, accuracy {:0.4f}".format(epoch, loss[-1],accuracy[-1]))

        if save: model_save(model, data_path, model_name, "latest")
        if not (epoch % 10):
            if save: model_save(model, data_path, model_name, epoch)

    if save: model_save(model, data_path, model_name, "final", echo=True, temp=False)

    return loss







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
plt.plot(h)





# Use embedding to predict next letter
# - Model 5_ used only 3000 obs, 5.1_ used 20000
# - model 5_ shows interesting pattern with k:
#       - k at the start followed by n
#       - ok is v.different

h=[]
model_name="5.1_next_letter"
raw_data, coder = get_raw_data(n_chars=20000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
num_flags = y.shape[2] - 1
model5 = create_model_D(hidden_units=100, max_len=max_len, num_categories=num_cats, embedding_size=50, num_flags=num_flags, model_name=model_name, mask_zero=True)
h += model_fit(model5, X, y, epochs=2, batch_size=5)
plt.plot(h)



# Stateful
h=[]
model_name="6_Stateful_next_letter"
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=3, flags=0)
num_flags = y.shape[2] - 1
model6 = create_model_E(hidden_units=10, num_flags=num_flags, model_name=model_name)
h += model_fit_stateful(model6, X, y, epochs=20)
plt.plot(h)








################ Predicting ##################

# Import a model
model1a = model_load("model_vowel1a_final")
model1a = model_load("Finals\model_vowel1a_final")
model1b = model_load("\Finals\model_vowel1b_final")
model3 = model_load("Finals\model_vowel3_final")
model4 = model_load("Finals\model_4.1_vowel_space_final")
model4b = model_load("Finals\model_4b_vowel_space_final")
model5 = model_load("model_5_next_letter_40")
model5 = model_load("Finals\model_5_next_letter_final")
model6 = model_load("Finals\model_6_Stateful_next_letter_final")


# Set model info to Model, x_dims, normalise (2 if using embeddings)
model_info = (model1, 3, False)
model_info = (model1b, 3, False)

model_info = (model3, 2, False)
model_info = (model4, 2, False)
model_info = (model4b, 2, False)
model_info = (model5, 2, False)
model_info = (model6, 3, False)




def predict(model_info, word, flag=1, max_len=10):
    raw_obs = list(word)
    model, x_dims, normalise = model_info

    x_data = createX([raw_obs], coder, max_len = max_len, normalise=normalise)
    x_shape = (1, max_len, 1)[0:x_dims]
    X = x_data.reshape(x_shape)

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

def predict_next_letter(model_info, text, ax=None, top_n=None, max_len=10, sorting=False):
    '''
    :param model_info:
    :param letters:
    :param prefix:
    :param max_len:
    :param sorting:
    :param flag: Which category to give the probability for.  If None then returns the argmax category
    :return:
    '''
    if ax is None:
        ax = plt.gca()
    else:
        ax.clear()

    raw_obs = list(text)
    prediction_index = len(text)-1
    pred = predict(model_info, raw_obs, flag=None, max_len=max_len)

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

def predict_next_letter_stateful(model_info, text, ax=None, top_n=None):
 #   text = 'hello'
#    top_n = 5

    if ax is None:
        ax = plt.gca()
    else:
        ax.clear()

    model, _, _ = model_info
    model.reset_states()
    for i, letter in enumerate(text):
        pred = predict(model_info, [letter], flag=None, max_len=1)


    # Keep 5 most likely next letters
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
plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax)
myUi.ChartUpdater(plot_Fn = plot_next_letter)


model6 = model_load("Finals\model_6_Stateful_next_letter_final")
model_info = (model6, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax)
myUi.ChartUpdater(plotFn = plot_next_letter)


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







