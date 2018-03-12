'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

#from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
import h5py
import matplotlib.pyplot as plt
import MyUtils.utils_nn as myNn
import MyUtils.utils_plot as myPlot
import math

# build the VGG16 network with ImageNet weights
model = vgg16.VGG16(weights='imagenet', include_top=False)
model.summary()

img = myNn.find_max_input(model, 'block5_conv1', 0, (128, 128))
plt.imshow(img)
plt.show()

images = list(map(lambda x: myNn.find_max_input(model, 'block5_conv1', x, (128, 128)) , range(5)))

images = [img]
myPlot.display_images(images)
plt.show()

