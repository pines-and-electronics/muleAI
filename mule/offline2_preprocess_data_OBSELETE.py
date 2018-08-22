raise USE JUPYTER

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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% IO
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)

#%% Get a dataset
THIS_DATASET = '20180807 201756'
df_records = pd.read_pickle(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'))

hist = df_records['steering_signal'].hist()
hist = df_records['throttle_signal'].hist()

frames=np.load(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'))

#%%
def get_record(df_records, frames, idx):
    this_frame = frames[idx]
    this_steering = df_records[df_records['timestamp'] == idx]['steering_signal']
    #this_steering = df_records[idx]
    return this_frame,this_steering

##indices= sel_indices
def get_n_records(df_records, frames, indices):
    #this_frame = np.array[frames[idx] for idx in indices]
    these_frames = [frames[idx] for idx in indices]
    
    frame_array = np.stack([frames[idx] for idx in indices], axis=0)
    #this_steering = df_records[df_records['timestamp'] == idx]['steering_signal']
    these_steering = df_records[df_records['timestamp'].isin(indices)]['steering_signal'].values
    #this_steering = df_records[idx]
    return frame_array,these_steering


frame, steering = get_record(df_records,frames, '1533666134582')

sel_indices = df_records.sample(5)['timestamp'].values
sel_frames, sel_steerings = get_n_records(df_records, frames, sel_indices)

#%%
plt.figure(1)
for f, s in zip(sel_frames, sel_steerings):
    imgplot = plt.imshow(f)
    plt.xticks([])
    plt.yticks([])
    print(f,s)
#%% RANDOMIZE FOR TESTING
if 1:
    df_records.loc[:,'steering_signal'] = np.random.uniform(low=-1, high=1, size=len(df_records.loc[:,'steering_signal']))

#%% Get y_labels
#y = df_records.loc[:,['steering_signal','throttle_signal']].values
y = df_records.loc[:,['steering_signal']].values
#y = df_records.loc[:,['steering_signal','throttle_signal']].values
    
def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr

#a = el[0]

#a= -1
#%%
binned_y= list()
for i,el in df_records.loc[:,['steering_signal',]].iterrows():
    binned = linear_bin(el[0])
    #print(el, binned)
    binned_y.append(binned)
#%% 
binned_y_matrix = np.array(binned_y)

#%%
y = df_records.loc[:,['steering_signal',]].values

#%% Get X_train from zip
def get_X_train(this_selected_data):
    npy_records = list()
    npz_file=np.load(this_selected_data['camera_numpy_zip'])
    for k in npz_file:
        #print(k)
        #print(npz_file[k])
        npy_records.append(npz_file[k])
    X_train = np.array(npy_records)
    return X_train

X_train = get_X_train(selected_data)

#%%
del df_records