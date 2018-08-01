import sys
import glob,os
import json
import pandas as pd
#import tensorflow as tf
import logging
import zipfile
#import re
#import datetime
import cv2
import numpy as np

#%% IO
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)

#%% Generate JPG for feedback
#this_selected_data = selected_data
def write_jpg(this_selected_data):
    path_npz = this_selected_data['camera_numpy_zip']
    
    arrays = np.load(path_npz)
    timestamps = [k for k in arrays.keys()]
    timestamps.sort()
    
    # Create a directory for the JPEGs
    path_jpg = os.path.join(this_selected_data['this_dir'], 'jpg')
    if not os.path.exists(path_jpg):
        os.mkdir(path_jpg)
    
    # Print to .jpg
    for k in timestamps:
        #print(k, arrays[k].shape)
        img = arrays[k]
        arrays[k]
        out_path = os.path.join(path_jpg,'{}.jpg'.format(k))
        cv2.imwrite(out_path, img)
    logging.debug("Wrote {} .jpg to {}".format(len(timestamps),path_jpg))
    
    #cv2.imshow('test',img)

write_jpg(selected_data)

#dataset_def['json_size_MB']


#%% Get the JSON records
def get_records(this_selected_data):
    path_json = this_selected_data['json_record_zip']
    json_records = list()

    with zipfile.ZipFile(path_json, "r") as f:
        json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
        
        for json_file in json_file_paths:
            this_fname = os.path.splitext(json_file)[0] 
            this_timestep = this_fname.split('_')[1]
            d = f.read(json_file)
            d = json.loads(d.decode("utf-8"))
            d['timestamp'] = this_timestep
            json_records.append(d)
    logging.debug("Returning {} json records from {}".format(len(json_file_paths),this_selected_data['json_record_zip']))
    
    return pd.DataFrame(json_records)    
df_records = get_records(selected_data)


#%% RANDOMIZE FOR TESTING
if 1:
    df_records.loc[:,'steering_signal'] = np.random.uniform(low=-1, high=1, size=len(df_records.loc[:,'steering_signal']))

#%% Get y_labels
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