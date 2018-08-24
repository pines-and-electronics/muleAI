import os

from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from glob import iglob

import tensorflow as tf
import numpy as np

import cv2
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
assert 'ffmpeg' in mpl.animation.writers.list(), "Need to install ffmpeg!"

%matplotlib inline

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML

import time


#%%