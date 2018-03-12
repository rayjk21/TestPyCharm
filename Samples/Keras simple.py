from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


batch_size = 128
nb_classes = 10
nb_epoch = 2

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


binary=True
if not binary: # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
else: # my modification as binary classification
    c = 1
    Y_train = (y_train==c).astype(np.int32)
    Y_test = (y_test==c).astype(np.int32)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

if not binary: 
    model.add(Dense(10))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

else:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms)


model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

if binary:
    esti = model.predict(X_test)
    estiy = (esti>0.5)
    acc = np.mean(estiy[:,0] == Y_test)
    print ('my acc:', acc)

















#######################################################################################
# Simple MLP 
#######################################################################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)


print("X:", x_train.shape)
print("Y:", y_train.shape)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)



















# Styles for using Keras:

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# 
# model = Sequential([
#     Dense(32, input_shape=(784,)),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])
# 
# 





# model = Sequential()
# model.add(Dense(32, input_dim=784))
# model.add(Activation('relu'))





#    ### START CODE HERE ###
#    # Feel free to use the suggested outline in the text above to get started, and run through the whole
#    # exercise (including the later portions of this notebook) once. The come back also try out other
#    # network architectures as well. 
#    
#    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
#    X_input = Input(input_shape)
#
#    # Zero-Padding: pads the border of X_input with zeroes
#    X = ZeroPadding2D((3, 3))(X_input)
#
#    # CONV -> BN -> RELU Block applied to X
#    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
#    X = BatchNormalization(axis = 3, name = 'bn0')(X)
#    X = Activation('relu')(X)
#
#    # MAXPOOL
#    X = MaxPooling2D((2, 2), name='max_pool')(X)
#
#    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
#    X = Flatten()(X)
#    X = Dense(1, activation='sigmoid', name='fc')(X)
#
#    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
#    model = Model(inputs = X_input, outputs = X, name='HappyModel')
#
#    ### END CODE HERE ###

