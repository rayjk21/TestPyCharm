
from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K

from keras.callbacks import ModelCheckpoint
import numpy as np
import keras.utils as Kutils
import keras.preprocessing.sequence as K_prep_seq

import matplotlib.pyplot as plt




# See https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
# 1 = Naive LSTM for Learning One-Char to One-Char Mapping
# 2 = Naive LSTM for a Three-Char Feature Window to One-Char Mapping
# 3 = Naive LSTM for a Three-Char Time Step Window to One-Char Mapping
# 4 - LSTM State WITHIN A Batch  (artificial example)
# 5 - Stateful LSTM for a One-Char to One-Char Mapping
# 6 - LSTM with Variable-Length Input to One-Char Output





data_path = r"C:\Temp\TestPyCharm\Data\Models\001 RNN abc"
alphabet = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_abc_data(seq_length = None, max_length=None, num_inputs=100, alphabet=alphabet):
    # define the raw dataset
    vocab_length = len(alphabet)
    # create mapping of characters to integers (0-25) and the reverse
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []

    if seq_length is not None:
        num_inputs = len(alphabet) - seq_length
        for i in range(0, num_inputs, 1):
            seq_in = alphabet[i:i + seq_length]
            seq_out = alphabet[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
            print (seq_in, '->', seq_out)

    if max_length is not None:
        for i in range(num_inputs):
            start = np.random.randint(len(alphabet) - 2)
            end = np.random.randint(start, min(start + max_length, len(alphabet) - 1))
            sequence_in = alphabet[start:end + 1]
            sequence_out = alphabet[end + 1]
            dataX.append([char_to_int[char] for char in sequence_in])
            dataY.append(char_to_int[sequence_out])
            print(sequence_in, '->', sequence_out)

    return dataX, dataY, vocab_length, char_to_int, int_to_char



def pad_X(dataX, maxlen):
    # convert list of lists to array and pad sequences if needed
    X = K_prep_seq.pad_sequences(dataX, maxlen=maxlen, dtype='float32')

    return X

### Reshape the data
def reshape_X(dataX, time_steps, n_features):
    '''

    :param dataX:
    :param dataY:
    :param vocab_length:
    :return: X, y
    '''

    X = np.array(dataX)
    n_samples = X.shape[0]

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_samples, time_steps, n_features))

    print("X shape {}".format(X.shape))

    return X


### Encode data
def transform_data(dataX, dataY, vocab_length):
    # normalize
    X = dataX / float(vocab_length)

    # one hot encode the output variable
    y = Kutils.to_categorical(dataY)

    return X,y






####################  Create models   #################################

def model_simple(input_shape, hidden_units=32):
    # if you will be feeding data 1 character at a time your input shape should be (31, 1) since your input has 31 timesteps, 1 character each

    # create and fit the model
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_stateful(batch_size, input_shape, hidden_units=32):
    # if you will be feeding data 1 character at a time your input shape should be (31, 1) since your input has 31 timesteps, 1 character each

    # create and fit the model
    model = Sequential()
    model.add(LSTM(hidden_units, batch_input_shape=(batch_size, input_shape[0], input_shape[1]), stateful=True))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_fit(model, X, y, epochs, batch_size, stateful=False, shuffle=True):
    # When running as stateful, the whole training set is the single large sequence, so must not shuffle it.
    # When not stateful, each item in the training set is a different individual sequence, so can shuffle these
    if stateful:
        shuffle = False
        lbl = "Iteration"
    else:
        lbl = "Epoch"

    accs = []
    loss = []
    for i in range(epochs):
        # if the shuffle argument in model.fit is set to True (which is the default),
        # the training data will be randomly shuffled at each epoch
        h = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=shuffle).history

        # When not stateful, state is reset automatically after each input
        # When stateful, this is suppressed, so mush manually reset after the epoch (effectively the one big sequence)
        if stateful: model.reset_states()
        if not(i % 10): print("{} {:4d} : loss {:.04f}, accuracy {:0.4f}".format(lbl, i,h['loss'][0], h['acc'][0]))
        accs += h['acc']

    accs += h['acc']
    loss += h['loss']
    return accs



def evaluate(model, data, decode_dict, time_steps, n_features, vocab_length=26, max_len = None, random=False):
    dataX, X, y = data
    # summarize performance of the model
    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # demonstrate some model predictions
    for i in range(0, 25):
        if random: ix = np.random.randint(len(dataX))
        else: ix = i
        inputX = dataX[ix]
        if max_len:
            inputX = K_prep_seq.pad_sequences([inputX], maxlen=max_len, dtype='float32')[0]
        x = np.reshape(inputX, (1, time_steps, n_features))
        x = x / float(vocab_length)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        prob = np.max(prediction)
        result = decode_dict[index]
        seq_in = [decode_dict[i] for i in inputX]
        print ("{} -> {} (p={:0.4f})".format(seq_in, result, prob))


def evaluate_stateful(model, data, decode_dict, batch_size, time_steps, n_features, vocab_length=26, random=False):
    dataX, X, y = data

    # summarize performance of the model
    model.reset_states()
    scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # Must reset otherwise is set on predicting Z
    model.reset_states()

    # Pick starting letter
    if random:
        seed = np.random.randint(vocab_length)
    else:
        seed = 0

    for i in range(0, vocab_length-1):
        x = np.reshape(seed, (1, time_steps, n_features))
        x = x / float(vocab_length)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        print (decode_dict[seed], "->", decode_dict[index])
        # Next input is the previous output
        seed = index









################## 1 = Naive LSTM for Learning One-Char to One-Char Mapping ######################


seq_length = 1

# Input Shape :
time_steps = seq_length
n_features = 1

# Number of Input observations to read at a time, before updating weights
batch_size = 1

dataX, dataY, vocab_length, encode_dict, decode_dict = get_abc_data(seq_length=seq_length)
X = reshape_X(dataX, time_steps = time_steps, n_features = n_features)
X, y = transform_data(X, dataY, vocab_length)

h = []
model1 = model_simple(input_shape=(X.shape[1], X.shape[2]))
h += model_fit(model1, X, y, epochs=300, batch_size=batch_size)
plt.plot(h)
evaluate(model1, (dataX, X, y), decode_dict, time_steps, n_features, vocab_length)

# We can see that this problem is indeed difficult for the network to learn.
# The reason is, the poor LSTM units do not have any context to work with.
# Each input-output pattern is shown to the network in a random order and the state of the network is reset after each pattern
# (each batch where each batch contains one pattern).
# This is abuse of the LSTM network architecture, treating it like a standard multilayer Perceptron.
# Next, let’s try a different framing of the problem in order to provide more sequence to the network from which to learn.





################### 2 = Naive LSTM for a Three-Char Feature Window to One-Char Mapping ###################

seq_length = 3

# Input Shape : Presenting as multiple features in a single timestep (wrong)
time_steps = 1
n_features = seq_length

# Number of Input observations to read at a time, before updating weights
batch_size = 1

dataX, dataY, vocab_length, encode_dict, decode_dict = get_abc_data(seq_length=seq_length)
X = reshape_X(dataX, time_steps = time_steps, n_features = n_features)
X, y = transform_data(X, dataY, vocab_length)

model2 = model_simple(input_shape=(X.shape[1], X.shape[2]))
h = []
h += model_fit(model2, X, y, epochs=500, batch_size=batch_size)
evaluate(model2, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length)

# We can see a small lift in performance that may or may not be real.
# This is a simple problem that we were still not able to learn with LSTMs even with the window method.
# Again, this is a misuse of the LSTM network by a poor framing of the problem.
# Indeed, the sequences of letters are time steps of one feature rather than one time step of separate features.
# We have given more context to the network, BUT NOT MORE SEQUENCE as it expected.
# In the next section, we will give more context to the network in the form of time steps






################ 3 = Naive LSTM for a Three-Char Time Step Window to One-Char Mapping ###################

# In Keras, the intended use of LSTMs is to provide context in the form of time steps, rather than windowed features like with other network types.
# We can take our first example and simply change the sequence length from 1 to 3.

seq_length = 3

# Presenting as single feature over multiple timesteps (right)
time_steps = seq_length
n_features = 1

# Number of Input observations to read at a time, before updating weights
batch_size = 1


dataX, dataY, vocab_length, encode_dict, decode_dict = get_abc_data(seq_length=seq_length)
X = reshape_X(dataX, time_steps = time_steps, n_features = n_features)
X, y = transform_data(X, dataY, vocab_length)

model3 = model_simple(input_shape=(X.shape[1], X.shape[2]))
h = []
h += model_fit(model3, X, y, epochs=500, batch_size=batch_size)

evaluate(model3, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length)
evaluate(model3, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length, random=True)

# We can see that the model learns the problem perfectly as evidenced by the model evaluation and the example predictions.
# But it has learned a SIMPLER PROBLEM. Specifically, it has learned to predict the next letter from a sequence of three letters in the alphabet.
# It can be shown any random sequence of three letters from the alphabet and predict the next letter.
# It can not actually enumerate the alphabet.

# The LSTM networks are stateful. They should be able to learn the whole alphabet sequence,
# but by default the Keras implementation RESETS THE NETWORK STATE after each training batch.






################# 4 - LSTM State WITHIN A Batch (artificial example) ###################

# The Keras implementation of LSTMs resets the state of the network after each input. But it holds the state between timesteps within one input.
# This suggests that if we had a batch size large enough to hold all input patterns and if all the input patterns were ordered sequentially,
# that the LSTM could use the context of the sequence within the batch to better learn the sequence.

# We can demonstrate this easily by modifying the first example for learning a one-to-one mapping and
# increasing the batch size from 1 to the size of the training dataset.

# Additionally, Keras shuffles the training dataset before each training epoch.
# To ensure the training data patterns remain sequential, we can disable this shuffling.

# The network will learn the mapping of characters using the the within-batch sequence,
# but this context will not be available to the network when making predictions.
# We can evaluate both the ability of the network to make predictions randomly and in sequence.

seq_length = 1

# Back to original naive approach v1
time_steps = seq_length
n_features = 1

# Number of Input observations to read at a time, before updating weights
# This is a misuse of the batch size (later will make stateful)
# - effectively turning a batch into an ovservation
# - since this is the whole data, there is only one observation!
batch_size = len(dataX)

dataX, dataY, vocab_length, encode_dict, decode_dict = get_abc_data(seq_length=seq_length)
X = reshape_X(dataX, time_steps = time_steps, n_features = n_features)
X, y = transform_data(X, dataY, vocab_length)

# NB. Fit model for more epochs
model4 = model_simple(input_shape=(X.shape[1], X.shape[2]), hidden_units=16)
h = []
h += model_fit(model4, X, y, epochs=5000, batch_size=batch_size, shuffle=False)

evaluate(model4, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length)
evaluate(model4, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length, random=True)


# As we expected, the network is able to use the within-sequence context to learn the alphabet, achieving 100% accuracy on the training data.
# Importantly, the network can make accurate predictions for the next letter in the alphabet for randomly selected characters. Very impressive.









#################### 5 - Stateful LSTM for a One-Char to One-Char Mapping ####################

# We have seen that we can break-up our raw data into FIXED SIZE SEQUENCES and that this representation can be learned by the LSTM,
# but only to learn random mappings of 3 characters to 1 character.
# We have also seen that we can PERVERT BATCH SIZE to offer more sequence to the network, but only during training.

# Ideally, we want to expose the network to the ENTIRE SEQUENCE and let it learn the inter-dependencies,
# rather than us define those dependencies explicitly in the framing of the problem.

# We can do this in Keras by making the LSTM layers STATEFUL and manually resetting the state of the network at the end of the epoch,
# which is also the end of the training sequence.  # This is truly how the LSTM networks are INDENDED TO BE USED.
# We find that by allowing the network itself to learn the dependencies between the characters, that we need a smaller network (half the number of units) and fewer training epochs (almost half).

# We first need to define our LSTM layer as stateful. In so doing, we must EXPLICITLY specify the batch size as a dimension on the INPUT shape.
# This also means that when we evaluate the network or make predictions, we must also specify and ADHERE to this same batch size.
# This is not a problem now as we are using a batch size of 1.
# This could introduce DIFFICULTIES when making predictions when the batch size is not one as predictions will need to be made in batch and in sequence

seq_length = 1

time_steps = seq_length
n_features = 1

batch_size = 1


dataX, dataY, vocab_length, encode_dict, decode_dict = get_abc_data(seq_length=seq_length)
X = reshape_X(dataX, time_steps = time_steps, n_features = n_features)
X, y = transform_data(X, dataY, vocab_length)

model5 = model_stateful(batch_size, input_shape=(X.shape[1], X.shape[2]), hidden_units=32)
h = []
h = model_fit(model5, X, y, epochs=1000, batch_size=batch_size, stateful=True)
plt.plot(h)

# Must reset before using model, as prediction depends on the current state
model5.reset_states()
evaluate_stateful(model5, (dataX, X, y), decode_dict, batch_size, time_steps, n_features, vocab_length)
# Has just learned to repeat the alphabet!  Not predict the next letter
evaluate_stateful(model5, (dataX, X, y), decode_dict, batch_size, time_steps, n_features, vocab_length, random=True)

# We can see that the network has memorized the entire alphabet perfectly.
# It used the context of the samples themselves and learned whatever dependency it needed to predict the next character in the sequence.
# We can also see that if we seed the network with the first letter, that it can correctly rattle off the rest of the alphabet.
# We can also see that it has only learned the full alphabet sequence and that from a cold start.
# When asked to predict the next letter from “K” that it predicts “B” and falls back into regurgitating the entire alphabet.
# To truly predict “K” the state of the network would need to be warmed up iteratively fed the letters from “A” to “J”.
# This tells us that we could achieve the same effect with a “stateless” LSTM by preparing training data like








##################  6 - LSTM with Variable-Length Input to One-Char Output  #######################

# In the previous section, we discovered that the Keras “STATEFUL” LSTM was really only a SHORTCUT TO REPLAYING the first n-sequences,
# but didn’t really help us learn a generic model of the alphabet.

# In this section we explore a variation of the “stateless” LSTM that learns RANDOM SUBSEQUENCES of the alphabet and
# an effort to build a model that can be given arbitrary letters or subsequences of letters and predict the next letter in the alphabet.

# Firstly, we are changing the framing of the problem. To simplify we will define a MAXIMUM INPUT sequence LENGTH
# and set it to a small value like 5 to speed up training.
# This defines the maximum length of subsequences of the alphabet will be drawn for training.
# In extensions, this could just as set to the full alphabet (26) or longer if we allow looping back to the start of the sequence.

num_inputs = 500
max_len = 5

time_steps = max_len
n_features = 1

batch_size = 1
h = []

dataX, dataY, vocab_length, encode_dict, decode_dict = get_abc_data(max_length=max_len, num_inputs = num_inputs)
X = reshape_X(pad_X(dataX, max_len), time_steps = time_steps, n_features = n_features)
X, y = transform_data(X, dataY, vocab_length)



# NB. Fit model for fewer epochs as lots more in samples in dataset
model6 = model_simple(input_shape=(X.shape[1], X.shape[2]), hidden_units=32)
h += model_fit(model6, X, y, epochs=100, batch_size=batch_size)
plt.plot(h)

evaluate(model6, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length, max_len)
evaluate(model6, (dataX, X, y), decode_dict,  time_steps, n_features, vocab_length, max_len, random=True)


















###########################  Sequence of Words example #########################


input_words = "The quick brown fox jumped over the lazy dog."


def read_words(filename=None, n=10000):
    if filename is None : filename = "ptb.train.txt"
    filepath = data_path + "\\" + filename
    print("Reading filename: {}".format(filepath))
    with tf.gfile.GFile(filepath, "r") as f:
        return f.read(n).replace("\n", " ")

def prep_word_seq(all_words = input_words):
    def words_to_letters(words=input_words):
        return list(words)

    def build_vocab(itemlist):
        counter = collections.Counter(itemlist)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        items, _ = list(zip(*count_pairs))
        item_to_id = dict(zip(items, range(len(items))))

        return item_to_id

    def itemlist_to_ids(itemlist, item_to_id):
        return [item_to_id[item] for item in itemlist if item in item_to_id]

    # Convert all input words to seq of letters
    all_letters = words_to_letters(all_words)

    train_letters = all_letters
    valid_letters = all_letters
    test_letters  = all_letters

    item_to_id = build_vocab(all_letters)
    vocabulary = len(item_to_id)

    train_data = itemlist_to_ids(train_letters, item_to_id)
    valid_data = itemlist_to_ids(valid_letters, item_to_id)
    test_data = itemlist_to_ids(test_letters, item_to_id)

    id_to_item = dict(zip(item_to_id.values(), item_to_id.keys()))

    print("{} = {}".format(train_letters[:5], train_data[:5]))
    print(item_to_id)
    print(vocabulary)
    print(" ".join([id_to_item[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, id_to_item, item_to_id

### Get input word data
ptbdata = read_words("ptb.train.txt", n=10000)
train_data, valid_data, test_data, vocab_length, decode_dict, encode_dict = prep_word_seq(ptbdata)


####################  Data Generators (optional)   #################################

class KerasBatchGenerator(object):

    '''
        num_steps – this is the number of words that we will feed into the time distributed input layer of the network
        skip_steps is the number of words we want to skip over between training samples within each batch
    '''

    def __init__(self, data, num_steps, batch_size, vocab_length, skip_step=None):
        if skip_step is None: skip_step = num_steps
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab_length = vocab_length
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocab_length))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = Kutils.to_categorical(temp_y, num_classes=self.vocab_length)
                self.current_idx += self.skip_step
            yield x, y

def create_generators(train_data, valid_data, vocab_length, num_steps, batch_size):
    train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocab_length)
    valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocab_length)
    return train_data_generator, valid_data_generator



####################  Create  & Run model   #################################

def model_2layer_embedding(num_steps, vocab_length, hidden_size = 10, use_dropout=True):
    model = Sequential()
    model.add(Embedding(vocab_length, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocab_length)))
    model.add(Activation('softmax'))

    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model

def run_model_with_generators(model, train_gen, valid_gen, num_epochs=5):
    # Number of iterations to run for each epoch.
    # The calc ensures that the whole data set is run through the model in each epoch
    def calc_steps(gen):
        return len(gen.data) // (gen.batch_size * gen.num_steps)

    print("Using training data of length {}".format(len(train_gen.data)))

    valid_steps = calc_steps(valid_gen)
    train_steps = calc_steps(train_gen)

    checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

    model.fit_generator(train_gen.generate(), train_steps, num_epochs,
                        validation_data=valid_gen.generate(),
                        validation_steps=valid_steps, callbacks=[checkpointer])

    model.save(data_path + "\\final_model.hdf5")



num_steps = 8
batch_size = 1
train_data_generator, valid_data_generator = create_generators(train_data, valid_data, vocab_length, num_steps=num_steps, batch_size=batch_size)
model = model_2layer_embedding(num_steps, vocab_length)
print(model.summary())

run_model_with_generators(model, train_data_generator, valid_data_generator, num_epochs=5)





################# Make Predictions  ###################################

model = load_model(data_path + "\\model-05.hdf5")

dummy_iters = 40
example_training_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocab_length, skip_step=1)

print("Training data:")
for i in range(dummy_iters):
    dummy = next(example_training_generator.generate())


# Length of prediction sequence
num_predict = 8
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "
for i in range(num_predict):
    data = next(example_training_generator.generate())
    prediction = model.predict(data[0])
    predict_word = np.argmax(prediction[:, num_steps-1, :])
    true_print_out += decode_dict[train_data[num_steps + dummy_iters + i]] + " "
    pred_print_out += decode_dict[predict_word] + " "
print(true_print_out)
print(pred_print_out)


def code(letters):
    codes = []
    for l in letters:
        codes.append(encode_dict[l])
    return codes

codes = code(list("abcdefgh"))
prediction = model.predict(codes)
predict_word = np.argmax(prediction[:, num_steps - 1, :])