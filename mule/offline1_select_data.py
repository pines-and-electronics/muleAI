import sys
import glob,os
import json
import pandas as pd
#import tensorflow as tf
# Check versions
#assert tf.__version__ == '1.8.0'

import logging
import zipfile
#import re
import datetime
#import cv2
import cv2
import shutil

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


#%% IO
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
#LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE_DATA2'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)

#%%
#this_dir = LOCAL_PROJECT_PATH

def confirm_numpy(numpy_zip):
    # Look inside
    with zipfile.ZipFile(numpy_zip, "r") as f:
        fnames = (os.path.splitext(name) for name in f.namelist())
        timestamps, extensions = zip(*fnames)
        assert all(ext == '.npy' for ext in extensions)
        #[ts.isoformat() for ts in timestamps]
        
        #ts = timestamps[0]
        datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
        datetime_stamps.sort()
        return datetime_stamps

#json_zip = dataset_def['json_record_zip']
def confirm_json(json_zip):
    # Look inside
    with zipfile.ZipFile(json_zip, "r") as f:
        fnames = (os.path.splitext(name) for name in f.namelist())
        timestamps, extensions = zip(*fnames)
        assert all(ext == '.json' for ext in extensions)
        #[ts.isoformat() for ts in timestamps]
 
        timestamps = [ts.split('_')[1] for ts in timestamps]
        #ts = timestamps[0]
        datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
        datetime_stamps.sort()
        return datetime_stamps

def process_time_steps(dataset_def):
    # Get the record timestamps
    numpy_timestamps = confirm_numpy(dataset_def['camera_numpy_zip'])
    json_timestamps = confirm_json(dataset_def['json_record_zip'])
    
    # Ensure timestamp alignment
    assert numpy_timestamps == json_timestamps, "Temporal alignment failure"
    
    # Analysis of timesteps
    timestamps = pd.Series(numpy_timestamps)
    
    dataset_def['num_records'] =  len(timestamps)
    
    dataset_def['elapsed_time'] = timestamps.iloc[-1] - timestamps.iloc[0]
    dataset_def['elapsed_time_mins'] = dataset_def['elapsed_time'].total_seconds() / 60
    
    # Analysis of delta-times
    ts_deltas = (timestamps-timestamps.shift()).fillna(0)
    stats = ts_deltas[0:-1].describe()
    
    dataset_def['ts_deltas_mean'] = stats['mean'].total_seconds() * 1000
    dataset_def['ts_deltas_std'] = stats['std'].total_seconds() * 1000
    return dataset_def

def process_datetime(dataset_def):
    dataset_def['this_dt_string'] = os.path.split(dataset_def['this_dir'])[1]
    dataset_def['this_dt'] = datetime.datetime.strptime(dataset_def['this_dt_string'], '%Y%m%d %H%M%S')
    dataset_def['this_dt_iso'] = dataset_def['this_dt'].isoformat()
    dataset_def['this_dt_nice'] = dataset_def['this_dt'].strftime("%A %d %b %H:%M")
    return dataset_def

def process_camera_zip(dataset_def):
    dataset_def['camera_numpy_zip'] = glob.glob(os.path.join(dataset_def['this_dir'],'camera_numpy.zip'))[0]
    assert os.path.exists(dataset_def['camera_numpy_zip'])
    dataset_def['camera_size_MB'] = os.path.getsize(dataset_def['camera_numpy_zip'])/1000/1000
    return dataset_def

def process_json_records(dataset_def):
    dataset_def['json_record_zip'] = glob.glob(os.path.join(dataset_def['this_dir'],'json_records.zip'))[0]
    dataset_def['json_size_MB'] = os.path.getsize(dataset_def['json_record_zip'])/1000/1000
    #dataset_def['num_json'] = os.path.getsize(dataset_def['json_record_zip'])/1000/1000
    return dataset_def

def process_jpg_zip(dataset_def):
    if os.path.exists(os.path.join(dataset_def['this_dir'],'jpg_images.zip')):
        dataset_def['flg_jpg_zip_exists'] = True
        dataset_def['jpg_zip'] = glob.glob(os.path.join(dataset_def['this_dir'],'jpg_images.zip'))[0]
        dataset_def['jpg_zip_size_MB'] = os.path.getsize(dataset_def['jpg_zip'])/1000/1000
        #dataset_def['num_json'] = os.path.getsize(dataset_def['json_record_zip'])/1000/1000
        
        
        # Look inside
        with zipfile.ZipFile(dataset_def['jpg_zip'], "r") as f:
            #fnames = (os.path.splitext(name) for name in f.namelist())
            dataset_def['num_jpgs'] = len(f.namelist())
    else: 
        dataset_def['flg_jpg_zip_exists'] = False
        
        
    return dataset_def

#
#def create_jpgs(dataset_def):
#    dataset_def['folder_jpg'] = os.path.join(dataset_def['this_dir'],'jpg')
#    if not os.path.exists(dataset_def['folder_jpg']):
#        dataset_def['num_jpgs'] = 0
#    else:
#        dataset_def['num_jpgs'] = len(glob.glob(os.path.join(dataset_def['folder_jpg'],'*.jpg')))
#    
#    return dataset_def
#


#%% Zip the files
    
#def make_archive(self, source, destination):
#    base = os.path.basename(destination)
#    name = base.split('.')[0]
#    format = base.split('.')[1]
#    archive_from = os.path.dirname(source)
#    archive_to = os.path.basename(source.strip(os.sep))
#    #print(source, destination, archive_from, archive_to)
#    shutil.make_archive(name, format, archive_from, archive_to)
#    shutil.move('%s.%s'%(name,format), destination)
#    
#    logging.debug("Created archive {}".format(destination))


#%% Generate JPG for feedback

def write_jpg(this_data_def):
    path_npz = this_data_def['camera_numpy_zip']
    
    arrays = np.load(path_npz)
    timestamps = [k for k in arrays.keys()]
    timestamps.sort()
    
    # Create a directory for the JPEGs
    path_jpg = os.path.join(this_data_def['this_dir'], 'jpg')
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
    return path_jpg
    #cv2.imshow('test',img)

def zip_jpgs(path_jpg, target_path):
    jpg_files = glob.glob(os.path.join(path_jpg,'*.jpg'))
    
    with zipfile.ZipFile(target_path, 'w') as myzip:
        for f in jpg_files:
            name = os.path.basename(f)
            myzip.write(f,name)
            os.remove(f)
    logging.debug("Zipped {} to {}".format(len(jpg_files),target_path))

    
    
def delete_jpgs(path_jpg):
    jpg_files = glob.glob(os.path.join(path_jpg,'*.jpg'))
    
    # Remove all .npy files, confirm
    [os.remove(f) for f in jpg_files]
    
    jpg_files = glob.glob(os.path.join(path_jpg,'*.jpg'))
    assert len(jpg_files) == 0
    os.rmdir(path_jpg)
    logging.debug("Deleted all .jpg files".format())
    
#%% PROCESS ALL DATASETS!

this_data_dir = LOCAL_PROJECT_PATH
def get_datasets(this_data_dir):
    directoy_list = glob.glob(os.path.join(this_data_dir,'*'))
    
    # Iterate over each directory
    dataset_def_list = list()
    for i,this_dir in enumerate(directoy_list):
        
        dataset_def = dict()
        dataset_def['this_dir'] = this_dir
        
        # Date time from directory name
        dataset_def = process_datetime(dataset_def)
        
        # Numpy camera arrays
        dataset_def = process_camera_zip(dataset_def)
        
        # json state records
        dataset_def = process_json_records(dataset_def)
        
        # JPG zip
        dataset_def = process_jpg_zip(dataset_def)
        
        # If the JPG zip doesn't exist, create it
        if not dataset_def['flg_jpg_zip_exists']:
            # JPG images
            #dataset_def = create_jpgs(dataset_def)
    
            # Write the JPG's
            #if not dataset_def['num_jpgs']:
            path_jpg = write_jpg(dataset_def)
                # Reload the jpg definition!
             #   dataset_def = process_jpgs(dataset_def)
            
            # Zip them
            path_zip = os.path.join(dataset_def['this_dir'],'jpg_images.zip')
            zip_jpgs(path_jpg,path_zip)
            delete_jpgs(path_jpg)
            
            #dataset_def = zip_jpgs(dataset_def)

            dataset_def = process_jpg_zip(dataset_def)
            
        #raise
        # Time step analysis
        dataset_def = process_time_steps(dataset_def)

        assert dataset_def['flg_jpg_zip_exists'], "jpg zip does not exist, {}".format(dataset_def)
        
        assert dataset_def['num_jpgs'] == dataset_def['num_records']
        
        logging.debug("Dataset {}: /{}, recorded on {}".format(i, dataset_def['this_dt_string'],dataset_def['this_dt_nice']))
        
        # Summary of this dataset
        logging.debug("{} aligned records found over {:0.2f} minutes.".format(
                dataset_def['num_records'],
                dataset_def['elapsed_time_mins']
                ))

        logging.debug("Timestep analysis: {:0.0f} +/- {:0.0f} ms".format(
                      dataset_def['ts_deltas_mean'],
                      dataset_def['ts_deltas_std'],
                      ))
        
        # Append
        dataset_def_list.append(dataset_def)
        
    # Done
    this_df_datasets = pd.DataFrame(dataset_def_list)
    this_df_datasets = this_df_datasets.sort_values(['this_dt']).reset_index(drop=True)        
    return  this_df_datasets

df_datasets = get_datasets(LOCAL_PROJECT_PATH)

#%%
def select_data(this_df):
    row_str =  "{:<4} {:<20} {:<30} {:<12} {:<6.0f}"
    head_str = "{:<4} {:<20} {:<30} {:<12} {:<6}"
    
    fields = ('this_dt_string','this_dt_nice','num_records','camera_size_MB')
    head = ['idx'] + [*fields]
    print(head_str.format(*head))
    for i,r in this_df.iterrows():
        this_row_str = [r[fld] for fld in fields]
        print(row_str.format(i,*this_row_str))
    
    ds_idx = int(input("Select dataset number:") )
    this_dataset = this_df.iloc[ds_idx]
    return this_dataset
selected_data = select_data(df_datasets)

#%%
#del (df_datasets)


#%%
if 0:
    
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

