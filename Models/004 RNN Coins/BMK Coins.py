
import numpy as np
import pandas as pa
import pandas as pd
import aptecoPythonUtilities.utils_explore as my_exp
import matplotlib.pyplot as plt

import MyUtils.Embeddings as my_emb
import MyUtils.utils_base as my
import MyUtils.utils_prep as my_prep
import MyUtils.utils_nn as my_nn


model_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Models\004 RNN Coins"






##############   Raw Data


def get_raw_data(data_info=None):

    raw = pa.read_csv(r"D:\FastStats\PUBLISH\BMK\project\All Transactions.csv")
    # type(raw['TransDate'][0])

    # Slow to parse all the dates, so convert later
    # raw = pa.read_csv(r"D:\FastStats\PUBLISH\BMK\project\All Transactions.csv", parse_dates=['TransDate'], dayfirst=True)

    # my_exp.overview(raw)
    # my_exp.detail(raw)

    # Keep by volume
    # my_prep.freq_hist(raw, "Product")
    sub = my_prep.freq_cut(raw, "Product", 20, 1000, show=False)[["CustId", "Product", "TransDate"]]

    #my_prep.freq_hist(sub, "Product")
    #my_prep.freq_hist(sub, "CustId")
    #my_exp.overview(sub)
    #my_exp.detail(sub)

    sub2 = my_prep.freq_cut(sub, "Product", 100, 1000, show=False)
    #my_exp.overview(sub2)
    #my_exp.detail(sub2)

    # Convert dates for just the kept volumes
    df = sub2.copy()
    df['TransDt'] = pa.to_datetime(df['TransDate'], dayfirst=True)
    df.sort_values(by=['CustId', 'TransDt'], inplace=True)

    # Check dates have parsed correctly (2018-02-12 showld be 12th Feb)
    #print(type(df['TransDt'].iloc[0]))
    #print(df[df['CustId'] == 285158].dropna)


    # Group by cust
    date_lines = my_prep.toGroupsDf(df, 'CustId', 'TransDt')
    return date_lines



def time_line_plot(lines):

    def set_x_axis(ax, min, max):
        # min = pa.Timestamp('2016-01-20')
        # max = pa.Timestamp('2018-02-20')
        import matplotlib.dates as mdates

        if (min is None) or (max is None): return

        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y')

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years...
        min_year = np.datetime64(min, 'Y')
        max_year = np.datetime64(max, 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(min_year, max_year)

    def show_quarters(ax, min, max):
        min_year = pa.Timestamp(np.datetime64(min, 'Y'))
        max_year = pa.Timestamp(np.datetime64(max, 'Y') + np.timedelta64(1, 'Y'))

        for q in my.daterange(min_year, max_year, quarters=2):
            w = 90 # days
            ax.barh(left=q, width=w, y=0, height=1,  align='edge', color='grey')

    def timeplot(ts, ax, show_xlabels=False, min=None, max=None):
        ts = np.array(ts)

        set_x_axis(ax, min, max)
        show_quarters(ax, min, max)

        # Draw actual lines
        ax.vlines(ts, [0], [1])

        if (not show_xlabels): ax.set_xticklabels([], visible=False)
        ax.set_yticklabels([], visible=False)
        ax.get_yaxis().set_ticks([])

    def print_values(i, timeline):
        print ("Timeline {}: ".format(i), end="")
        ts = sorted(timeline)
        for t in ts: print(t.date(), end=", ")
        print("\n")

    def multiple_lines(timelines, print=False):
        if (type(timelines) == pa.DataFrame):
            timelines = timelines.iloc[:,0]
        if (type(timelines) == pa.Series):
            timelines = timelines.values
        num_lines = len(timelines)
        all_points = np.hstack(timelines)
        min = np.min(all_points)
        max = np.max(all_points)

        for i in range (num_lines):
            ax = plt.subplot(num_lines, 1, i+1)
            ts = timelines[i]
            # Set labels on last plot
            show_xlables = True if (i==num_lines-1) else False
            timeplot(ts, ax, show_xlables, min, max)
            if print: print_values(i, timelines[i])

        plt.show()

    multiple_lines(lines)

def get_lengths(lines, plot=False):
    f = my_prep.vectorizeA(lambda line: np.shape(line))
    lengths = np.squeeze(f(lines))
    if plot:
        print(lengths[0:10])
        plt.hist(lengths)
    return lengths




##########  X, Y Data #################################

def dates_diff(dt_line , start):
    '''
    Returns multiple values for each point on the date line:
    (Converts array of DateTimes to an array of integer-arrays)
    - each integer array is the [difference from 'start', day of month]
    :param dt_line:
    :param start:
    :return:
    '''
    def date_diff2(t): return [(t-start).days, t.day]

    f = my_prep.vectorizeR(date_diff2)
    diff_line = f(dt_line)
    return diff_line


def time_line_gaps(dt_line):
    '''
        Returns the differences (gaps) between a line of dates
    '''

    diff = np.diff(dt_line)
    f = my_prep.vectorizeA(lambda t: t.days)
    diff_days = f(diff)
    # Insert 0 at start
    diff0 = np.insert(diff_days, 0, 0)
    return diff0

def shift_line(line):
    from scipy.ndimage.interpolation import shift
    return shift(line, [-1], mode='constant', cval=0)



def get_xy_data(raw_data , data_info=None, n_time=10, remove_zeros=True, start=None, printing=True):
    # Make start date Now if none is set
   # if start is None: start = pa.Timestamp.now()
   # if type(start) is str: start = pa.Timestamp(start)

    if data_info is not None:
        n_time       = data_info.get('n_time')       or n_time
        remove_zeros = data_info.get('remove_zeros') or remove_zeros
        print("n_time:{}".format(n_time))
        print("remove_zeros:{}".format(remove_zeros))

    # Find the gaps between the datelines
    lines = raw_data['TransDt'].values

    if printing: print("Calculating gaps between transactions")
    f = my_prep.vectorizeA(lambda line: time_line_gaps(line))
    lines = f(lines)

    if remove_zeros:
        lines = my_prep.remove_zeros(lines, printing)


    # Cut up into samples if longer than length, or pad out with zeros
    X = np.array(list(my_prep.samples_of_length(lines, n_time, printing)))

    f = my_prep.vectorizeA(shift_line)
    Y = f(X)

    # Could also find the difference from start
    return X, Y


def checkXY(X, Y, i=0, n=3):
    for i in range(i, i+n):
        print()
        print(X[i])
        print(Y[i])



############################################################################################

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Input, Flatten, Dropout, TimeDistributed, BatchNormalization, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
import keras.utils
import keras



def set_shape(M, dims, label="", printing=False):
    m_shape = M.shape
    if (len(m_shape)==1):
        m_reshape = [1, m_shape[0], 1][0:dims]
    if (len(m_shape)==2):
        m_reshape = [m_shape[0], m_shape[1], 1][0:dims]

    if (m_shape != m_reshape):
        if printing: print("Reshaping {} from {} to {}".format(label, m_shape, m_reshape))
        M = M.reshape(m_reshape)
    return M

def create_model_1(model_info=None, n_time=20, hidden_units=10, embedding_size=10, dropout=0.2, mask_zero=True, model_name='Temp_Model_1' ):
    '''
        Input has to be 2 dimensions: n_obs * n_time_stamp (with no n_features)
        Output is categorical
    :return:
    '''
    print("Creating model 1")
    data_info       = model_info.get('data_info')
    if data_info:
        n_time          = data_info.get('n_time') or n_time


    hidden_units    = model_info.get('hidden_units')    or hidden_units
    embedding_size  = model_info.get('embedding_size')  or embedding_size
    dropout         = model_info.get('dropout')         or dropout
    mask_zero       = model_info.get('mask_zero')       or mask_zero
    model_name      = model_info.get('model_name')      or model_name

    model = Sequential(name=model_name)
    model.add(LSTM(hidden_units, input_shape=(n_time,1), return_sequences=True, stateful=False))
    model.add(Dense(1, name="Output"))
    model.compile(loss=keras.losses.mse, optimizer='adam', metrics=['accuracy'])

    model_info.update({'model':model})

    print("Created model: {}".format(model_name))
    print(model.summary())
    return model

def model_fit(model_or_model_info, X, Y, epochs, batch_size=8, stateful=False, shuffle=True, save=True, x_dims=3, y_dims=3, model_path=model_path):
    import time

    if (type(model_or_model_info) is dict):
        model_info = model_or_model_info
        data_info  = model_info.get('data_info')
        model      = model_info.get('model')
        model_path = model_info.get('model_path')   or model_path
        model_name = model_info.get('model_name')   or model.name
        batch_size = model_info.get('batch_size')   or batch_size
        stateful   = model_info.get('stateful')     or stateful
    else:
        model = model_or_model_info
        model_name = model.name


    # When running as stateful, the whole training set is the single large sequence, so must not shuffle it.
    # When not stateful, each item in the training set is a different individual sequence, so can shuffle these
    if stateful:
        shuffle = False
        batch_size = 1
        lbl = "Iteration"
        timesteps = X.shape[1]
        if (timesteps != 1):
            raise ValueError("When using stateful it is assumed that each X value has a single time-step but there are {}".format(timesteps))
    else:
        lbl = "Epoch"

    if data_info:
        x_dims = data_info.get('x_dims') or x_dims
        y_dims = data_info.get('x_dims') or y_dims

    X = set_shape(X, x_dims,"X")
    Y = set_shape(Y, y_dims,"Y")


    print("Fitting model '{}' over {} epochs with batchsize {}".format(model_name,epochs,batch_size))
    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(Y.shape))
    print()

#    metrics = n_hot_accuracy_metric()
    precision   = []
    accuracy = []
    loss = []
    for epoch in range(epochs):
        # if the shuffle argument in model.fit is set to True (which is the default),
        # the training data will be randomly shuffled at each epoch
        h = model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0, shuffle=shuffle
                      , validation_split=0.25
                      #  ,callbacks=[metrics]
                      ).history

        # Got error on callback with dropout i
        print("{} {:4d} : loss {:.04f}, accuracy {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], time.ctime()))
        #print("{} {:4d} : loss {:.04f}, accuracy {:0.4f}, Precision {:0.4f} - {}".format(lbl, epoch, h['loss'][0], h['acc'][0], h['MyPrecision'][0], time.ctime()))
        accuracy += h['acc']
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





def predict(Xt, model_info=None, n_time=15, output='last'):
    '''

    :param Xt:
    :param model_info:
    :param n_time:
    :param output: 'last' or 'full'
    :return:
    '''
    if model_info:
        model       = model_info.get('model')
        data_info   = model_info.get('data_info')
        n_time      = data_info.get('n_time') or n_time

    Xt = my_prep.pad(Xt, n_time)
    Xt = set_shape(Xt, 3)

    pred = model.predict(Xt, verbose=0)
    f = np.vectorize(lambda x: round(x))
    pred = my_prep.remove_zeros(f(pred), printing=False)

    pred = np.squeeze(pred)
    if (output=='last'):
        last = my_prep.vectorizeA(lambda line: line[-1])
        pred = last(pred)
    return pred

def predict_rank(X, model_info, n=100):
    Xt = X[0:n]
    Yt = predict(Xt, model_info)
    Xs = my_prep.remove_zeros(Xt, printing=False)
    df = pa.DataFrame({'Timeline':list(Xs), 'Next':Yt })
    df.sort_values(by='Next', inplace=True, ascending=False)
    return df[df['Next']>0]

def plot_gaps(gaps):
    offset = (lambda x: pa.Timestamp.now() - np.timedelta64(np.asscalar(x),'D'))
    offsetV = my_prep.vectorizeA(offset)
    offsetVV = my_prep.vectorizeA(offsetV)

    gaps_to_points = my_prep.vectorizeA(lambda gaps: gaps[::-1].cumsum()[::-1])
    points = gaps_to_points(gaps)
    times = offsetVV(points)
    #print(gaps)
    #print(points)
    #print(times)
    time_line_plot(times)





####################   Using ModelInfo to specify models   ##################################

raw_data = get_raw_data()
time_line_plot(raw_data[0:10])
get_lengths(raw_data['TransDt'], plot=True)



def data_with(data_settings):
    data_info = {'n_time':20, 'remove_zeros':True, 'x_dims':3  }
    data_info.update(data_settings)
    return data_info

def model_with(model_settings):
    model_info = {'create_fn': create_model_1, 'model_name':"default_model", 'model_path':model_path,
                  'stateful':False,
                  'dropout':0.0, 'hidden_units':10, 'embedding_size':50,
                  'loss':keras.losses.mse,
                  'mask_zero':True}

    model_info.update(model_settings)
    return model_info

def prep_data(model_or_data_info, load=False):
    if (model_or_data_info.get('data_info') is None):
        data_info = model_or_data_info
    else:
        data_info = model_or_data_info['data_info']

    if load: use_data = get_raw_data(data_info)
    else: use_data = raw_data

    X, Y = get_xy_data(use_data, data_info=data_info)

    return X,Y,raw_data

def build_model(model_info, data_only = False, load=False, prep=True, create=False, fit=True, epochs=10):
    if load | prep | data_only:
        X, Y, raw_data = prep_data(model_info, load)
        if data_only:
            return X, Y, raw_data


    if create:
        print("Creating new model {}".format(model_info['model_name']))
        model = model_info['create_fn'](model_info)
    else:
        print ("Reusing existing model {}".format(model_info['model_name']))
        model = model_info['model']

    if fit:
        model_fit(model_info, X, Y, epochs=epochs, batch_size=5)
        #model.evaluate(X, Y, batch_size=5)


data_info1 = data_with({'n_time':15})

model_info1 = model_with({'create_fn' : create_model_1, 'model_name':"model1", 'data_info':data_info1, 'batch_size':50})
model_info1b = model_with({'create_fn': create_model_1, 'model_name':"model1b", 'data_info':data_info1, 'batch_size':50})




X,Y,raw_data = prep_data(model_info1)
checkXY(X, Y)
get_lengths(X, plot=True)




build_model(model_info1b, prep=True, create=True, fit=True, epochs=10)


my_nn.model_load(model_info1, suffix='final')






Xt = [
    [1,2,3],
    [100,100,100],
    [1,2,3,100,200],
    [10,20,30,40,50,60]
]


pred = predict_rank(X, model_info1, n=100)
print(pred)

low  = pred['Timeline'][-20:-10].values
hi  = pred['Timeline'][0:10].values
hi_low = np.hstack([hi, low])
plot_gaps(hi_low)













##############################################################################


















##############################################################################



