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
#import matplotlib
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import tensorflow as tf
from tensorflow.python import keras as ks
import sklearn as sk

import cv2

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


#%% Turn off plotting
# First. change the mode to GUI window output
%matplotlib qt
# Then disable output
plt.ioff()

#%% Turn on plotting
plt.ion()
%matplotlib inline

    
# In[9]:


# Conversion between categorical and floating point steering
def linear_bin(a):
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


def bin_Y(Y):
    d = [ linear_bin(y) for y in Y ]
    return np.array(d)


def unbin_Y(Y):
    d = [ linear_unbin(y) for y in Y ]
    return np.array(d)


#print()

#%%
##indices= sel_indices
def get_n_records(df_records, frames, indices):
    raise
    """
    """
    #this_frame = np.array[frames[idx] for idx in indices]
    these_frames = [frames[idx] for idx in indices]
    
    frame_array = np.stack([frames[idx] for idx in indices], axis=0)
    #this_steering = df_records[df_records['timestamp'] == idx]['steering_signal']
    these_steering = df_records[df_records['timestamp'].isin(indices)]['steering_signal'].values
    
    these_throttle = df_records[df_records['timestamp'].isin(indices)]['throttle_signal'].values
    
    timestamps = df_records[df_records['timestamp'].isin(indices)]['timestamp'].values
    #this_steering = df_records[idx]
    
    these_ts = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
    return frame_array,these_steering,these_throttle, these_ts
    

#frame, steering = get_record(df_records,frames, '1533666134582')


#%%




#%%
def get_full_records(this_frames, this_df_records, this_indices):
    #assert type(this_indices) == list or type(this_indices) == np.ndarray
    """Given a list of indices (timestamps), return a list of "Records""
    
    A record is used for further visualization or analysis of a given timestep.
    
    A Record is a convenience dictionary with following keys: 
    
        frames          : The frame images as a numpy array, directly from NPZ
        steering_signal : The steering at these times as a float
        throttle        : The throttle at these times as a float
        timestamp       : The timestep as a datetime object
    
        AND OPTIONALLY: 
        df_records['steering_pred_signal'] : The model predicted steering at these times
    
    """
    records = list()
    for this_idx in this_indices:
        #print(this_idx)
        rec = dict()
        rec['frame'] = this_frames[this_idx]
        rec['steering_signal'] = df_records.loc[this_idx]['steering_signal']
        #print(rec['steer'])
        rec['throttle'] = df_records.loc[this_idx]['throttle_signal']
        rec['timestamp_raw'] = df_records.loc[this_idx]['timestamp']
        #print()
        rec['timestamp'] = datetime.datetime.fromtimestamp(int(rec['timestamp_raw'])/1000)
        if 'steering_pred_signal' in df_records.columns:
            #'steering_signal' in df_records.columns
            rec['steering_pred_signal'] = df_records.loc[this_idx]['steering_pred_signal']

        records.append(rec)
    logging.debug("Created {} record dictionaries".format(len(this_indices)))
    
    return records
    #record = dict()
    #record['frame'] 
#     OLD METHOD, VECTORIZED:
#         rec['frame'] = np.stack(this_frames[this_idx] for idx in this_idx], axis=0)
#         rec['steer'] = df_records[df_records['timestamp'].isin(this_idx)]['steering_signal'].values
#         rec['throttle'] = df_records[df_records['timestamp'].isin(this_idx)]['throttle_signal'].values
#         timestamp = df_records[df_records['timestamp'].isin(this_idx)]['timestamp'].values
#         rec['timestamp'] = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamp]
#         rec['steer_pred'] = y_pred_floats[y_pred_floats.index.isin(this_idx)]['steering_pred'].values 



#%%
def get_predictions(model, frames_npz, df_records):
    """Augment the df_records with the predictions
    """
    df_records['steering_pred_cats'] = pd.Series(dtype=object)
    #df_records['steering_pred_argmax'] = 
    for i,idx in enumerate(df_records.index):
        
        this_frame = np.expand_dims(frames_npz[idx],0)
        this_frame.shape
        this_pred = model.predict(this_frame)
        df_records.loc[idx,'steering_pred_cats'] = [this_pred]
        df_records.loc[idx,'steering_pred_argmax'] = np.argmax(df_records.loc[idx,'steering_pred_cats'])
        df_records.loc[idx,'steering_pred_signal'] = linear_unbin(df_records.loc[idx,'steering_pred_cats'][0])
        if i%100 == 0:
            print(i)
    return df_records

#%%
    
def get_fig_as_npy(this_fig):
    raise
    """Take a matplotlib.figure.Figure and return it as a static npy RGB array 
    """
    canvas = mpl.backends.backend_agg.FigureCanvas(this_fig)
    canvas.draw() 
    width, height = this_fig.get_size_inches() * this_fig.get_dpi()
    width = int(width)
    height = int(height)
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return img


#%%
def plot_frames(records):
    """
    Render N records to analysis
    """
    fig=plt.figure(figsize=[20,18],facecolor='white')
    ROWS = 1
    COLS = 4
    NUM_IMAGES = ROWS * COLS

    for i,rec in enumerate(records):

        steer_cat = ''.join(['|' if v else '-' for v in linear_bin(rec['steering_signal'])])
        timestamp_string = rec['timestamp'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['timestamp'].microsecond))
        
        this_label = "{}\n{:0.2f} steering\n{:0.2f} throttle".format(rec['timestamp'],rec['steering_signal'],rec['throttle'])
        y = fig.add_subplot(ROWS,COLS,i+1)
        y.imshow(rec['frame'])
        #plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        t = y.text(5,25,this_label,color='green',alpha=1)
        #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
        t.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))
        y.text(80,105,steer_cat,fontsize=30,horizontalalignment='center',verticalalignment='center',color='green')
        #plt.title()
        



#%%

#joined['steering_pred_cats'] = joined['steering_pred'].apply(linear_bin)
#joined['steering_pred_argmax'] = joined['steering_pred_cats'].apply(np.argmax)



def gen_one_record_frame(rec):
    """From a Record dictionary, create a single summary image of that timestep. 
    
    The figure has no border (full image)
    
    Show a data box with throttle and steering values. 
    Show also the predicted values, if available. 
    
    Show a steering widget to visualize the current steering signal. 
    Show also the predicted value, if available. 
    
    """
    font_label_box = {
        'color':'green',
        'size':16,
    }
    font_steering = {'family': 'monospace',
            #'color':  'darkred',
            'weight': 'normal',
            'size': 45,
            }
    SCALE = 50
    HEIGHT_INCHES = 160*2.54/SCALE
    WIDTH_INCHES =  120*2.54/SCALE
    fig = plt.figure(frameon=False,figsize=(HEIGHT_INCHES,WIDTH_INCHES))
    #fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rec['frame'])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    ######## The data box ########
    timestamp_string = rec['timestamp'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['timestamp'].microsecond))
    if 'steering_pred_signal' in df_records.columns:
        this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,rec['steering_signal'],rec['steering_pred_signal'],rec['throttle'])
    else: 
        this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string,rec['steering_signal'],rec['throttle'])
    t1 = ax.text(2,15,this_label,fontdict=font_label_box)
    t1.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))

    ######## The steering widget HUD ########
    # Steering HUD : Actual steering signal
    # Steering HUD: Predicted steering angle
    steer_actual = ''.join(['|' if v else '-' for v in linear_bin(rec['steering_signal'])])
    text_steer = ax.text(80,105,steer_actual,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
    
    if 'steering_pred_signal' in df_records.columns:
        steer_pred = ''.join(['◈' if v else ' ' for v in linear_bin(rec['steer_pred'])])
        text_steer_pred = ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
    
    return fig

#%%
def get_frames(path_frames,frame_ids):
    """From a list of ID strings, fetch the npy arrays from the zip. 
    """
    if type(frame_ids) == str:
        frame_ids = [frame_ids]
    npz_file=np.load(path_frames)
    frames_array = np.stack([npz_file[idx] for idx in frame_ids], axis=0)
    logging.debug("Returning frames array {}".format(frames_array.shape))
    return frames_array

#%% Data gen
class MuleDataGenerator(ks.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, path_frames, path_records, 
                 batch_size=32, dim=None, n_channels=None, n_classes=15, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path_frames = path_frames
        assert os.path.exists(self.path_frames)
        self.path_records = path_records
        assert os.path.exists(self.path_records)
        logging.debug("** Initialize datagen **".format())
        logging.debug("Frames stored at: {}".format(self.path_frames))
        logging.debug("Records stored at: {}".format(self.path_records))
        logging.debug("{} samples over batch size {} yields {} batches".format(len(list_IDs),
                                                                                   self.batch_size,
                                                                                   math.ceil(len(list_IDs)/self.batch_size),))
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    # GET A BATCH!
    def __getitem__(self, index): 
        'Generate one batch of data'
        
        logging.debug("Generating batch {}".format(index))
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data by selecting these IDs
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __get_npy_arrays(self,list_IDs_temp):
        """Open the npz file and load n frames into memory"""
        # This is a pointer to the file
        npz_file=np.load(self.path_frames)
        #for k in list_ID_temp:
        #    npy_records.append(npz_file[k])
        #X_train = np.array(npy_records)
        
        frames_array = np.stack([npz_file[idx] for idx in list_IDs_temp], axis=0)
        logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))
        
        return frames_array
    
    def __get_records(self,list_IDs_temp):
        
        # Load the saved records
        df_records = pd.read_pickle(self.path_records)
        # Set the index to match
        df_records.index = df_records['timestamp']
        # Subset
        this_batch_steering = df_records.loc[list_IDs_temp]
        
        steering_values = this_batch_steering['steering_signal'].values
        
        #print(steering_values)
        steering_records_array = bin_Y(steering_values)
        
        #df_categorical_steering = df_records['steering_signal']
        #
        
        
        #records_array = df_records[]
        logging.debug("Generating {} records {}:".format(steering_records_array.shape[0],steering_records_array.shape))
        return steering_records_array
        
        #raise
    
    def __data_generation(self, list_IDs_temp):
        """
        """
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)
        
        X = self.__get_npy_arrays(list_IDs_temp)
        y = self.__get_records(list_IDs_temp)
        
        # Generate data
        #for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            #y[i] = self.labels[ID]
        #    pass

        return X, y

