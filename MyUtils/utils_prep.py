
import pandas as pa
import numpy as np
import sklearn as skl
import keras.preprocessing.sequence as K_prep_seq
import matplotlib.pyplot as plt


def groupArrays(df, byVar, ofVar, presorted=True):
   # byVar, ofVar = ('CustId', 'TransDt')
    cols = df[[byVar,ofVar]]
    if presorted:
        keys,values=cols.values.T
    else:
        keys,values=cols.sort_values(byVar).values.T
    ukeys,index=np.unique(keys,True)
    arrays=np.split(values,index[1:])
    df2=pa.DataFrame({byVar:ukeys, ofVar:arrays})
    #df2=pa.DataFrame({byVar:ukeys, ofVar:[list(a) for a in arrays]})
    df2.set_index(byVar, inplace=True)
    return df2



def xPerY(df, ofVar, byVar, renameTo=None, sort=True):
    per = df[[ofVar, byVar]].drop_duplicates().groupby(byVar).agg({ofVar:['count']})
    if renameTo==None: renameTo = "{} Per {}".format(ofVar, byVar)
    per.columns=[renameTo]
    if sort: per = per.sort_values(by=renameTo, ascending=False)
    return per[renameTo]




def freq_hist(df, column, range=None):
    '''
    :param df:
    :param column:
    :param range:  range=[1,100]
    :return:
    '''
    df[column].value_counts().hist(range=range)
    plt.show()

def freq_cut(df, column, min, max, show=True):
    if show: freq_hist(df, column)
    keep = df[column].value_counts().where(lambda x: (x>=min) & (x<=max)).dropna()

    sub = df[df[column].isin(keep.keys())]
    if show: freq_hist(sub, column)
    return sub









def create_encoder(data1d):
    '''
    :param:
        data1d = 1 dimensional array of non-unique values
    :return:
        Creates a sklearn encoder to provide sequential codes for each unique value
        Use with:
            - coder.transform(['e'])
            - coder.inverse_transform(3)
    '''
    unique_items = np.unique(data1d)
    coder = skl.preprocessing.LabelEncoder()
    coder.fit(list(unique_items))
    n = len(coder.classes_)
    npreview = min(n, 20)
    print("Created {} classes, including {}".format(n, coder.classes_[:npreview]))

    return coder


def pad_data(data, max_len, value):
    return K_prep_seq.pad_sequences(data, maxlen=max_len, value=value, padding='post')


def set_n_hot(positions, n_obs=None, n_pos=None):
    '''
        Creates array (n_obs,n_pos) of zeros
        and sets multiple 1's according to the positions
    :param positions:
        positions = r_obs
        Array with n_obs rows, giving positions to set to 1
            cols = np.array([[1, 2, 3], [4, 4, 4], [5, 5, 5]])
            print(set_n_hot(cols))
    :param n_pos:
    :param n_obs:
    :return:
        [[0. 1. 1. 1. 0. 0.]
        [0. 0. 0. 0. 1. 0.]
        [0. 0. 0. 0. 0. 1.]]
    '''
    if n_pos is None: n_pos = np.max(positions) + 1
    if n_obs is None: n_obs = positions.shape[0]
    slots = np.zeros((n_obs, n_pos))
    col_ix = positions.T
    row_ix = np.arange(n_obs)
    slots[row_ix, col_ix] = 1
    return slots



def onehot_2D(a):
    ncols = a.max() + 1
    # Create 2D array, that will later be reshaped to 3D
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    # Set values: every row, just the columns from the input values
    rows = np.arange(a.size)
    cols = a.ravel()
    out[rows, cols] = 1
    # Extend shape to 3D
    out.shape = a.shape + (ncols,)
    return out

def rolling_3D(arr3D, window = 2, agg='max'):
    '''
    :param arr3D:       Array (n_obs, n_time, n_cats)
    :param window:      Size of rolling window
    :param agg:         Function to apply over timesteps ('max', 'sum', 'mean', 'count')
    :return:
    '''
    def roll(agg):
        if agg == 'max':
            combine = lambda out, t : np.maximum(out, t)
        elif agg == 'sum':
            combine = lambda out, t: np.add(out, t)
        elif agg == 'count':
            combine = lambda out, t: 0/0 # raise Exception("Need to aggregate counts")

        # Initialise ready to accumulate time slices
        out = np.zeros(arr3D.shape)
        t = arr3D
        # Accumulate the timesteps ahead
        for i in range(window):
            # Bring time slices forward and reset the last one to avoid rolling round
            t = np.roll(t, axis=1, shift=-1)
            t[:, -1, :] = 0
            out = combine(out, t)

        return out

    if agg == 'max':
        return roll('max')
    elif agg == 'sum':
        return roll('sum')
    elif agg == 'mean':
        return (roll('sum') / roll('count'))








def vectorizeA(fn):
    '''
    Takes a (scalar->scalar) function which is to be applied to each element of an array
    Returns a function that can be applied to an array
     - this function applies the given function to each element of the array
     - returning another array of the same shape
    :param fn:
    :return:
    '''
    def do_it (array):
        return np.array([fn(p) for p in array])
    return do_it


def vectorizeR(fn):
    '''
    Takes a (scalar->M-dim array) function which is to be applied to each element of an array
    Returns a function that can be applied to an N-dim array
     - this function applies the given function to the N elements of the array
     - returning another array of shape (N, M)
    :param fn:
    :return:
    '''
    def do_it (array):
        return np.row_stack((fn(p) for p in array))
    return do_it




def to_np_arrays(items):
    '''
    Recursively creates np.array of np.arrays from list of lists of ...
    :param items:
    :return:
    '''
    if (type(items) is not list):
        return items
    else:
        if (type(items[0]) is list):
            return np.array(list(map(to_np_arrays, items)))
        else:
            return np.array(items)




def samples_of_length(lines, length=10, stride=None):
    length=5
    if stride is None: stride = length
    from keras.preprocessing.sequence import pad_sequences

    def get_lengths(lines):
        f = vectorizeA(lambda line: np.shape(line))
        lengths = np.squeeze(f(lines))
        return lengths

    def pad(short):
        return (np.pad(short, (0, length - len(short)), 'constant'))
    def pad_shorts(shorts):
        for short in shorts:
            yield(pad(short))
    def sample_long(longs):
        for long in longs:
            for start in range(len(long)-length, 0, -stride):
                yield long[start:start+length]
                # If not enough left to make another full sample...
                if (start-stride < 0):
                    #... could output the unused bit padded with 0s
                   # yield(np.pad(long[0:start], (0,length-start), 'constant'))
                    #.. or output full length, reusing a few items
                    yield long[0:length]
    def izip():
        for s,l in zip(iter_short, iter_long):
            yield s
            yield l

    lengths = get_lengths(lines)
    shorties  = lines[lengths<=length]
    long_ones = lines[lengths>length]

    iter_short = pad_shorts(shorties)
    iter_long  = sample_long(long_ones)
    iterLS = izip()

    #iter_short.__next__()
    #iter_long.__next__()
    #iterLS.__next__()
    return iterLS




###########################################################################

def check():
    Xs = np.array([[1, 2, 3, 0], [4, 4, 4, 0], [2,2,2, 0]])
    X01 = onehot_2D(Xs)
    r = rolling_3D(X01, 2)
    print(r)
