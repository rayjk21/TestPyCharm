
import keras as K
import keras.utils
import numpy as np
import pandas as pa
import keras
from keras.layers import Dense, Activation, Embedding, Input, Flatten, Dropout, TimeDistributed, BatchNormalization, Reshape, Lambda, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.preprocessing import OneHotEncoder

import MyUtils.Embeddings as my_emb
import aptecoPythonUtilities.utils_explore as my_exp
import MyUtils.utils_nn as my_nn
import MyUtils.utils_prep as my_prep


########################################################################################

#### Data

def create_n_hot_rolling(X, y_window = 1):

    print("Getting Y categories with look ahead of {}".format(y_window))

    # Convert to 0ne-hot, resetting the first index which will be for padded timestepts
    X01 = my_prep.onehot_2D(X)
    X01[:,:,0] = 0

    # Aggregate over max_ahead timesteps, setting 1 if any contain each item
    y_ = my_prep.rolling_3D(X01, y_window, 'max')

    return y_

def get_trans(raw, urn, item, code, n_obs = None):
    '''
    raw =hRaw
    urn  = 'Person URN'
     item = 'Destination'
     code = 'Code'
     n_obs=100
    '''

    enc = my_prep.create_encoder(raw[item])
    raw[code] = enc.transform(raw[item])
    df1 = raw[[urn, item, code]]
    if n_obs:
        print("Using {} obs".format(n_obs))
        df1 = df1.loc[0:n_obs]

    # Create arrays of items per Urn
    trans = my_prep.toGroupsDf(df1, urn, code)

    # Find maximum number of items per Urn
    max_items = max(trans[code].map(len))
    #multi = trans[np.count_nonzero(trans, axis=1) > 4]


    print("Created {} transactions with maximum length {}".format(len(trans), max_items))

    return trans[code]

def get_xy(trans, n_obs=None, n_time = None, y_window=1):
    if n_obs is not None:
        trans = trans[0:n_obs]
    if n_time is None:
        # Find the maximum length of transaction sequence
        n_time = max(trans.map(len))

    X = my_prep.pad_data(trans, n_time, -1) + 1
    Y = create_n_hot_rolling(X, y_window)

    print("Created X with shape:{}".format(X.shape))
    print("Created Y with shape:{}".format(Y.shape))

    return X, Y

def check(X, Y):
    trans_lengths = pa.Series(np.apply_along_axis(np.count_nonzero, 1, X))
    print("Maximum length {}".format(max(trans_lengths)))
    l = 1
    for l in range(1, max(trans_lengths)):
        # first_of_length_l
        ix = trans_lengths[trans_lengths==l].index
        if (len(ix)>0):
            ix = ix[0]
            print("Length {} at obs {}".format(l, ix))
            print("X:\n{}".format(X[ix, ...]))
            print("Y:\n{}".format(Y[ix, ...]))
            print()

#### Model Process

def load_data(data_info, n_obs=None):

    get_trans_fn = data_info['get_trans_fn']
    n_obs        = data_info['n_obs']
    y_window     = data_info['y_window']
    max_len      = data_info['max_len']

    trans = get_trans_fn(n_obs=n_obs)
    X, Y = get_xy(trans, n_obs=n_obs, n_time=max_len, y_window=y_window)

    if data_info:
        data_info['n_time'] = X.shape[1]
        data_info['n_cats'] = Y.shape[2]

    return X,Y

class one_hot_accuracy_metric(keras.callbacks.Callback):
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

        # Get predicted category
        if (prob.ndim==2):
            pred  = np.where(prob > 0.5, 1, 0)
            average = 'binary'

        # Get single top category for prediction & y
        if (prob.ndim==3):
            pred = my_nn.argmax3d(prob)
            yV   = my_nn.argmax3d(yV)
            # Need to average over the categories
            average = 'micro'

        # Print first 4 observations
     #  print("X: {} {}".format(xV.shape, xV[:4]))
     #  print("Probs: {} {}".format(prob.shape, prob[:4]))
     #  print("Preds: {} {}".format(pred.shape, pred[:4]))
     #  print("Actual: {} {}".format(yV.shape, yV[:4]))

        def flatten_non_zeros(A, Z):
            A[Z==0] = -1
            A = A.flatten()
            return A[A != -1]

        # Mask out where zeros in the input
        pred_ = flatten_non_zeros(pred, xV)
        yV_   = flatten_non_zeros(yV, xV)

        # Compare as flattened list over all obs and timestamps
        precision = sklearn.metrics.precision_score(yV_, pred_, average=average)
        logs.update({'MyPrecision':precision})

class n_hot_accuracy_metric(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs):
        import sklearn
        #xV=x
       # yV=y
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


        # Get single top category for prediction, convert back to one-hot
        topPred = my_nn.argmax3d(prob)
        pred01  = my_prep.onehot_2D(topPred)
        average = 'micro'

        # See where the prediction matches one of the n-hot y values
        n_hit = np.multiply(pred01, yV)
        # Count how many of the n-categories were hits (at most 1, since only the topPred was taken)
        hit = np.apply_along_axis(np.sum, 2, n_hit)

        # Print first 4 observations
     #  n=1
     #  print("X: {} {}".format(xV.shape, xV[:n]))
     #  print("Probs: {} {}".format(prob.shape, prob[:n]))
     #  print("Preds: {} {}".format(pred01.shape, pred01[:n]))
     #  print("Actual: {} {}".format(yV.shape, yV[:n]))
     #  print("Hits: {} {}".format(hit.shape, hit[:n]))

        def flatten_non_zeros(A, Z):
            A[Z==0] = -1
            A = A.flatten()
            return A[A != -1]

        # Ignore hit/miss where zeros in the input
        hit_ = flatten_non_zeros(hit, xV)

        # Precision is just the proportion of top predictions that hit one of the n-hot actual values
        precision = np.mean(hit_)

        logs.update({'MyPrecision':precision})

def model_fit(model_or_model_info, X, y, epochs, batch_size=8, stateful=False, shuffle=True, save=True):
    import time

    if (type(model_or_model_info) is dict):
        model_info = model_or_model_info
        data_info  = model_info['data_info']
        model      = model_info['model']
        model_path = model_info['model_path']
        batch_size = model_info['batch_size']
        stateful   = model_info.get('stateful')
    else:
        model = model_or_model_info


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

    model_name = "Unknown Model"
    if model_info:
        x_shape = X.shape
        x_shape = [x_shape[0], x_shape[1], 1][0:data_info['x_dims']]
        X.reshape(x_shape)
        model_name = model.name



    print("Fitting model '{}' over {} epochs".format(model_name,epochs))
    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))
    print()

    metrics = n_hot_accuracy_metric()
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

        # Got error on callback with dropout i
        print("{} {:4d} : loss {:.04f}, accuracy {:0.4f}, Precision {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], h['MyPrecision'][0], time.ctime()))
        #print("{} {:4d} : loss {:.04f}, accuracy {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], time.ctime()))
        accuracy += h['acc']
        #precision += h['MyPrecision']
        loss += h['loss']

        # When not stateful, state is reset automatically after each input
        # When stateful, this is suppressed, so must manually reset after the epoch (effectively the one big sequence)
        if stateful: model.reset_states()

        if save: my_nn.model_save(model, model_path , model_name, "latest")

        if not (epoch % 10):
            if save: my_nn.model_save(model, model_path , model_name, epoch)


    if save: my_nn.model_save(model, model_path , model_name, "final", echo=True, temp=False)

    if model_info:
        model_info.update({'model':model})

    return precision

def build_model(model_info, data_only = False, extract=True, create=False, fit=True, epochs=100):
    if (extract or data_only):
        X, Y = load_data(model_info['data_info'])
        if data_only:
            return X, Y

    if create:
        print("Creating new model {}".format(model_info['model_name']))
        model = model_info['create_fn'](model_info)
        model_info['model'] = model
    else:
        print ("Reusing existing model {}".format(model_info['model_name']))
        model = model_info['model']

    if fit:
        model_fit(model_info, X, Y,epochs=epochs)
        model.evaluate(X, Y, batch_size=5)








###################################################################################
# Holidays
###################################################################################

models_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Models\003 RNN Bookings"


hRaw = pa.read_csv(r"S:\develop\Data\Holidays\Bookings for All People.csv")
my_exp.overview(hRaw)
my_exp.detail(hRaw)

#destEmb = my_emb.CreateFromDf(hRaw,'Person URN', "Destination")
#destEmb.plotAll()




def create_model_1(model_info):
    '''
        Input has to be 2 dimensions: n_obs * n_time_stamp (with no n_features)
        Output is categorical
    :return:
    '''
    data_info       = model_info['data_info']
    n_time          = data_info['n_time']
    n_cats_in       = data_info['n_cats']
    n_cats_out      = n_cats_in

    hidden_units    = model_info['hidden_units']
    embedding_size  = model_info['embedding_size']
    dropout         = model_info['dropout']
    mask_zero       = model_info['mask_zero']
    model_name      = model_info['model_name']

    model = Sequential(name=model_name)
    model.add(Embedding(input_dim=n_cats_in, input_length=n_time, output_dim=embedding_size, mask_zero=mask_zero))
    model.add(LSTM(hidden_units, return_sequences=True, dropout=dropout))
    model.add(TimeDistributed(Dense(n_cats_out, activation = "softmax")))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def get_hols_trans(n_obs=None):
    #n_obs=100
    trans = get_trans(hRaw, urn  = 'Person URN', item = 'Destination', code = 'Code', n_obs=n_obs)
    return trans

def data_with(data_settings):
    data_info = {'n_obs':1000, 'x_dims':2, 'get_trans_fn':get_hols_trans, 'max_len':None}
    data_info.update(data_settings)
    return data_info

def model_with(model_settings):
    model_info = {
        'model_name':"default_model", 'model_path':models_path,
        'dropout':0.2, 'hidden_units':100, 'embedding_size':50,
        'loss':keras.losses.categorical_crossentropy,
        'batch_size':8,
        'mask_zero':True}

    model_info.update(model_settings)
    return model_info






data_info_a   = data_with({'n_obs':10000, 'y_window':1})
data_info_b   = data_with({'n_obs':10000, 'y_window':3})
model1a_info = model_with({'model_name': "Model_Bookings_1a", 'create_fn':create_model_1, 'data_info':data_info_a})
model1b_info = model_with({'model_name': "Model_Bookings_1b", 'create_fn':create_model_1, 'data_info':data_info_b})


build_model(model1a_info, epochs=5, create=True)
build_model(model1b_info, epochs=50, create=True)

my_nn.model_load(model1a_info, suffix='Final', sub_folder='Keep')
my_nn.model_load(model1b_info, suffix='Final', sub_folder='Keep')



Xa, Ya = load_data(data_info_a)
check(Xa, Ya)
my_nn.pred_counts(model1a_info, Xa, Ya, n_find=3)
# With window of 1, top prediction found 32.37% as the next item

# n_Find         0         1         2         3         4+
# n_Top
# 0       0.605339  0.323768  0.044596  0.016898  0.009399
# 1       0.873913  0.099090  0.020998  0.004200  0.001800
# 2       0.979902  0.012699  0.004300  0.001800  0.001300



Xb, Yb = load_data(data_info_b)
check(Xb, Yb)
my_nn.pred_counts(model1b_info, Xb, Yb, n_find=3)
# With window of 3, top prediction found slightly more, 36.6% as next item
# - strangely, the %found in positions 2,3 are less, even though these items
#   would have been presented in the expected output

# n_Find         0         1         2         3       4+
# n_Top
# 0       0.612539  0.366063  0.013499  0.005299  0.0026
# 1       0.916608  0.055694  0.015098  0.008099  0.0045
# 2       0.981702  0.013399  0.003000  0.001400  0.0005




