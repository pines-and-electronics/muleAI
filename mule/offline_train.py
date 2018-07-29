import sys
import glob,os
import json
import pandas as pd
import tensorflow as tf
import logging
import zipfile
import re
import datetime

#%% LOGGING for Spyder! Disable for production. 
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.DEBUG)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)s - %(module)-15s - %(funcName)-15s - %(message)s"
#FORMAT = "%(asctime)s L%(levelno)s: %(message)s"
FORMAT = "%(asctime)s - %(funcName)-20s: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.critical("Logging started")

#logging.getLogger("tensorflow").setLevel(logging.WARNING)

#%% Check versions
assert tf.__version__ == '1.8.0'

#%% IO
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)

directoy_list = glob.glob(os.path.join(LOCAL_PROJECT_PATH,'*'))

# Iterate over each directory
dataset_dirs = dict()
for i,this_dir in enumerate(directoy_list):
    this_dt_string = os.path.split(this_dir)[1]
    this_dt = datetime.datetime.strptime(this_dt_string, '%Y%m%d %H%M%S')
    this_dt_iso = this_dt.isoformat()
    
    #print("")

res = [(this_dir,os.path.split(this_dir)[1]) for this_dir in directoy_list]
this = res[-1][1]



#%%
for entry in directoy_list:
    if os.path.isdir(entry):
        logging.debug("Directory {}".format(entry))
        this_zip = os.path.join(entry,'state.zip')
        assert os.path.exists(this_zip)
        with zipfile.ZipFile(this_zip, "r") as f:
            npy_files = [name for name in f.namelist() if os.path.splitext(name)[1] =='.npy']
            json_files = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
            other_files = [name for name in f.namelist()]
            assert len(npy_files) + len(json_files) + 1 == len(other_files)
            
            for 
            
            #for name in f.namelist():
            #    data = f.read(name)
            #    print(name)
        #print name, len(data), repr(data[:10])
        
    #    this_zip
        #print(entry)    
    else:
        raise
#%%

X_keys = ['cam/image_array']
y_keys = ['user/angle', 'user/throttle']


kl = KerasLinear()
