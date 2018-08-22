import sys
import glob,os
import json
import pandas as pd
#import tensorflow as tf
import logging
import zipfile
#import re
#import datetime
import numpy as np
import os
import glob
import matplotlib
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import tensorflow as tf
from tensorflow.python import keras as ks
import sklearn as sk


#%% Logging


class LoggerCritical:
    def __enter__(self):
        my_logger = logging.getLogger()
        my_logger.setLevel("CRITICAL")
    def __exit__(self, type, value, traceback):
        my_logger = logging.getLogger()
        my_logger.setLevel("DEBUG")


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

with LoggerCritical():
    logging.debug("test block")