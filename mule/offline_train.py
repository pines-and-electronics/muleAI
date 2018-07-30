import sys
import glob,os
import json
import pandas as pd
import tensorflow as tf
import logging
import zipfile
import re
import datetime
import cv2

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

def get_datasets(this_dir):

    directoy_list = glob.glob(os.path.join(this_dir,'*'))
    
    # Iterate over each directory
    dataset_def_list = list()
    for i,this_dir in enumerate(directoy_list):
        dataset_def = dict()
        # Time stamp
        dataset_def['this_dt_string'] = os.path.split(this_dir)[1]
        dataset_def['this_dt'] = datetime.datetime.strptime(dataset_def['this_dt_string'], '%Y%m%d %H%M%S')
        dataset_def['this_dt_iso'] = dataset_def['this_dt'].isoformat()
        dataset_def['this_dt_nice'] = dataset_def['this_dt'].strftime("%A %d %H:%M")
        
        # Get the state zip
        this_zip = os.path.join(this_dir,'state.zip')
        assert os.path.exists(this_zip)
        dataset_def['zip'] = os.path.join(this_dir,'state.zip')
        dataset_def['size'] = os.path.getsize(this_zip)/1000/1000
        
        # Look inside
        with zipfile.ZipFile(this_zip, "r") as f:
            #print(f)
            npy_files = [name for name in f.namelist() if os.path.splitext(name)[1] =='.npy']
            json_files = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
            other_files = [name for name in f.namelist()]
            assert len(npy_files) + len(json_files) + 1 == len(other_files)
            #print(len(other_files))
            
        dataset_def['num_records'] = len(npy_files)
        #print(dataset_def)
        dataset_def_list.append(dataset_def)
        #dataset_def_dict
    
    return  pd.DataFrame(dataset_def_list)
df = get_datasets(LOCAL_PROJECT_PATH)
#df.columns

#%%
row_str = "{:<4} {:<20} {:<12} {:<6.0f}"
head_str = "{:<4} {:<20} {:<12} {:<6}"

fields = ('this_dt_nice','num_records','size')
head = ['idx'] + [*fields]
print(head_str.format(*head))
for i,r in df.iterrows():
    this_row_str = [r[fld] for fld in fields]
    print(row_str.format(i,*this_row_str))

ds_idx = int(input("Select dataset number:") )
this_dataset = df.iloc[ds_idx]

#%%
npy_records = list()
json_records = list()
with zipfile.ZipFile(this_dataset['zip'], "r") as f:
    
    npy_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] =='.npy']
    json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
    
    for npy_file in npy_file_paths:
        data = f.read(npy_file)
        npy_records.append(np.frombuffer(data))
    for json_file in json_file_paths:
        d = f.read(json_file)
        d = json.loads(d.decode("utf-8"))
        json_records.append(d)

logging.debug("{} json and {} npy records loads".format(len(json_records),len(npy_records)))

#print(f)

#%%
del (d, data, dataset_def, dataset_def_list, df, directoy_list, ds_idx, fields, 
head, head_str, i, json_file, json_files, json_file_paths, this_row_str, npy_file, npy_file_paths, npy_files, 
other_files, r, row_str, this_dataset, this_dir, this_zip)

#%%

single_path = r"/home/batman/MULE DATA/20180730 230317/camera_numpy.zip"
this = np.load(single_path)


#%%
#zip_path = r"/home/batman/MULE DATA/TEST/camera_array_1532981378805.npy"
this2 = np.load(this_dataset['zip'])
for k in this2.keys():
    print(k)
    #print(this2[k].shape)
#this2.f

#%% 
for rec in npy_records:
    print(rec)

#%%
this_arr = npy_records[0]
this_arr.shape
this_arr.ndim
#this
np.reshape(this_arr, (120, 160,3))


np.reshape(this_arr, (2, -1))

