################################## Simple version ###############################
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
import keras as keras
import matplotlib.pyplot as plt
import MyUtils.utils_nn as myNn


######## Get data
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10

# Normalise input values and reshape as single channel image 28x28
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28,28,1))
x_test = x_test.reshape((len(x_test), 28,28,1))

# Convert output class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
plt.imshow(x_test[0].reshape(28,28))
input_shape = (28,28,1)



###########  Create CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy)
model.fit(x_train, y_train, epochs=1, batch_size=256, validation_data=(x_test, y_test))

myNn.confusion(model, x_test, y_test, n_cases=10)




model.summary()

img = myNn.find_max_input(model, 'dense_42', 2)
img = myNn.find_max_input(model, 'conv2d_2', 0)

plt.imshow(img.reshape(28,28))
plt.show()





















