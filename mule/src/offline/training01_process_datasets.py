"""
Manage and analyze the Data Set directory.
Iterate over each Data Set and check which data elements have been created. 

"""

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
import cv2
import shutil
import json
from tabulate import tabulate
import tqdm

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

#%% Data set class
class AIDataSet():
    """A single datafolder object
    
    Description text
    
    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe object, with the 'timestamp' column for indexing. 
    path_frames_npz : str
        Path to frames numpy zip object. Can be accessed directly by np.load.
    """

    
    def __init__(self,path_data,data_folder):
        # Check the data folder
        self.path_data = path_data
        assert os.path.exists(self.path_data)
        self.data_folder = data_folder
        self.path_dataset = os.path.join(self.path_data,self.data_folder)
        assert os.path.exists(self.path_dataset)
        
        logging.debug("Data set {}, recorded on {}".format(self.data_folder,self.datetime_string))

        # Check the raw records zip, load to DataFrame
        self.path_records_zip = os.path.join(self.path_dataset,"json_records.zip")
        assert os.path.exists(self.path_records_zip)   
        self.df = self.load_records_df()
        logging.debug("Records {}".format(len(self.df)))
        
        # Check the raw frames zip, no need to unzip
        self.path_frames_npz = os.path.join(self.path_dataset,"camera_numpy.zip")
        assert os.path.exists(self.path_frames_npz)
        frames_timestamps = self.get_frames_timesteps() 
        logging.debug("Frames npz is {:0.2f} MB, {} records".format(self.frames_size,len(frames_timestamps)))
        
        # Assert timestep alignment
        assert all(self.df['timestamp'] == frames_timestamps), "Misaligned timestamps"
        
        # JPG folder
        JPG_FOLDER_NAME = "jpg_images"
        self.path_jpgs_dir = os.path.join(self.path_dataset,JPG_FOLDER_NAME)

    # =============================================================================
    # Query
    # =============================================================================
    @property
    def datetime_string(self):
        dt_obj = datetime.datetime.strptime(self.data_folder, '%Y%m%d %H%M%S')
        return dt_obj.strftime("%A %d %b %H:%M")

    @property
    def datetime_string_iso(self):
        dt_obj = datetime.datetime.strptime(self.data_folder , '%Y%m%d %H%M%S')
        return dt_obj.isoformat()

    @property
    def frames_size(self):
        """Size of frames npz array in MB
        """
        print(self.path_frames_npz)
        return os.path.getsize(self.path_frames_npz)/1000/1000
    
    def get_frames_timesteps(self):
        """Get timestamps from zipped NPY files. Return sorted pd.Series. 
        """
        # Open zip
        with zipfile.ZipFile(self.path_frames_npz, "r") as f:
            # Get the file names
            fnames = (os.path.splitext(name) for name in f.namelist()) 
            # Split and save
            timestamps, extensions = zip(*fnames)
        assert all(ext == '.npy' for ext in extensions)
        
        # Convert to datetime
        #datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
        # SORT!
        #datetime_stamps.sort()
        
        # Sorted and reindexed! 
        return pd.Series(timestamps).sort_values().reset_index(drop=True)
    
    #def jpg_frames():
        
    
    # =============================================================================
    # Load into memory
    # =============================================================================
    def load_records_df(self):
        """Get DataFrame from zipped JSON records. Return sorted pd.DataFrame. 
        
        All record columns created
        Timestamp column added (mtime)
        Sort the DF on timestamp
        Reindex        
        """        
        json_records = list()
        with zipfile.ZipFile(self.path_records_zip, "r") as f:
            json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
            # Each record is a seperate json file
            for json_file in json_file_paths:
                this_fname = os.path.splitext(json_file)[0] 
                this_timestep = this_fname.split('_')[1]
                d = f.read(json_file)
                d = json.loads(d.decode("utf-8"))
                d['timestamp'] = this_timestep
                json_records.append(d)
        # Sorted and reindexed! 
        return pd.DataFrame(json_records).sort_values(by='timestamp').reset_index(drop=True)
    
    # =============================================================================
    # Timestep analysis and processing
    # =============================================================================
    def augment_df_datetime(self):
        def convert_datetime(x):
            return datetime.datetime.fromtimestamp(int(x)/1000)
            #return datetime.datetime.strptime(x, '%Y%m%d %H%M%S')
        self.df['datetime'] = self.df['timestamp'].apply(convert_datetime)
        logging.debug("Augmented df with 'datetime' column".format())
        
    def process_time_steps(self):
        """Analysis of timestamps. Add some attributes to the class. 
        """
        assert 'datetime' in self.df.columns
        # Analysis of timesteps
        self.elapsed_time = self.df['datetime'].iloc[-1] - self.df['datetime'].iloc[0]
        self.elapsed_time_min = self.elapsed_time.total_seconds() / 60
        
        # Analysis of delta-times
        ts_deltas = (self.df['datetime']-self.df['datetime'].shift()).fillna(0)
        stats = ts_deltas[0:-1].describe()
        
        self.ts_deltas_mean = stats['mean'].total_seconds() * 1000
        self.ts_deltas_std = stats['std'].total_seconds() * 1000
        
        logging.debug("{:0.2f} minutes elapsed between start and stop".format(self.elapsed_time_min))

        logging.debug("Timestep analysis: {:0.0f} +/- {:0.0f} ms".format(
                      self.ts_deltas_mean,
                      self.ts_deltas_std
                      ))        
    
    # =============================================================================
    # Plotting
    # =============================================================================
    def histogram_steering(self):
        fig=plt.figure(figsize=[10,5],facecolor='white')
        hist_steering = self.df['steering_signal'].hist()
        return hist_steering
   
    def histogram_throttle(self):
        fig=plt.figure(figsize=[10,5],facecolor='white')
        hist_throttle = self.df['throttle_signal'].hist()
        #plot_url = py.plot_mpl(fig)    
    # =============================================================================
    # Video
    # =============================================================================
    def gen_one_record_frame(self, ts_string_index=None, int_index=None, source_jpg_folder='jpg_images'):
        """From a Record dictionary, create a single summary image of that timestep. 
        
        The figure has no border (full image)
        
        Show a data box with throttle and steering values. 
        Show also the predicted values, if available. 
        
        Show a steering widget to visualize the current steering signal. 
        Show also the predicted value, if available. 
        
        """
        this_ts = self.df[t]
        # Settings ############################################################
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
        
        # Figure ##############################################################
        fig = plt.figure(frameon=False,figsize=(HEIGHT_INCHES,WIDTH_INCHES))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Main Image ##########################################################
        jpg_path = os.path.join(self.path_dataset,source_jpg_folder)
        assert os.path.exists(jpg_path)
        img = mpl.image.imread(jpg_path)
        ax.imshow(img)
        raise
        self.df
        
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
            steer_pred = ''.join(['◈' if v else ' ' for v in linear_bin(rec['steering_pred_signal'])])
            text_steer_pred = ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
        
        return fig    
    
    # =============================================================================
    # Process frames to JPG
    # =============================================================================
    def write_jpgs(self, overwrite = False):
        """Write JPGs to disk from numpy zip file
        
        """
        jpg_files = glob.glob(os.path.join(self.path_jpgs_dir,'*.jpg'))
        if len(jpg_files) == len(self.df) and not overwrite:
            logging.debug("{} jpg files already exist, skip unless overwrite=True".format(len(self.df)))
            return
        
        # Open zip
        arrays = np.load(self.path_frames_npz)
        timestamps = [k for k in arrays.keys()]
        timestamps.sort()
        
        # Create a directory for the JPEGs
        path_jpg = os.path.join(self.path_dataset, self.path_jpgs_dir)
        if not os.path.exists(path_jpg):
            os.mkdir(path_jpg)
        
        # Print to .jpg
        for k in tqdm.tqdm(timestamps):
            img = arrays[k]
            arrays[k]
            out_path = os.path.join(path_jpg,'{}.jpg'.format(k))
            cv2.imwrite(out_path, img)
        logging.debug("Wrote {} .jpg to {}".format(len(timestamps),path_jpg))
        #return path_jpg
    
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

#%%

LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)
data1 = AIDataSet(LOCAL_PROJECT_PATH,"20180829 194519")
data1.augment_df_datetime()
data1.process_time_steps()
data1.write_jpgs(overwrite=False)
#data1.histogram_steering()
#data1.histogram_throttle()
data1.gen_one_record_frame(0)


raise
#%%
# =============================================================================
# Check which files exist in all datasets
# =============================================================================
#proj_path = LOCAL_PROJECT_PATH
CHECK_FILES = [
    'jpg_images.zip',
    'camera_numpy.zip',
    'df_record.pck',
    'json_records.zip',
    'video.mp4',
     ]

def check_files_exist(proj_path,these_check_paths):
    """Check if files exist in the data directory. 
    Args:
        proj_path: Project data directory path
        these_check_paths: Relative paths of files to check

    Returns:
        df_checkfiles: A simple dataframe with the status of files. 
    """
    
    directoy_list = glob.glob(os.path.join(proj_path,'*'))
    
    # Iterate over each directory
    check_files = list()
    for i,this_dir in enumerate(directoy_list):
        this_check_dict = dict()
        this_check_dict['this_dt_string'] = os.path.split(this_dir)[1]
        this_check_dict['this_dir']  = this_dir
        for path in these_check_paths:
            this_check_dict[path] = os.path.exists(os.path.join(this_dir,path))
        check_files.append(this_check_dict)
    df_checkfiles = pd.DataFrame(check_files)
    
    df_checkfiles.set_index('this_dt_string',inplace=True)
    df_checkfiles.sort_index(inplace=True)
    return(df_checkfiles)

df_checkfiles = check_files_exist(LOCAL_PROJECT_PATH,CHECK_FILES)
print(tabulate(df_checkfiles[CHECK_FILES],headers="keys",disable_numparse=True))

#%%
# =============================================================================
# Utility: Get timesteps for alignment
# =============================================================================
def check_numpy(numpy_zip):
    """Get timestamps from zipped NPY files.
    """
    # Look inside
    with zipfile.ZipFile(numpy_zip, "r") as f:
        fnames = (os.path.splitext(name) for name in f.namelist())
        timestamps, extensions = zip(*fnames)
    assert all(ext == '.npy' for ext in extensions)
    datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
    datetime_stamps.sort()
    return datetime_stamps

def check_json(json_zip):
    """Get timestamps from zipped JSON records.
    """
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

# =============================================================================
# Utility: Get .npy zip size
# =============================================================================
def check_camera_zip(this_dir):
    """Get the size of the stored camera arrays. 
    """
    return_dict = dict()
    return_dict['camera_numpy_zip'] = glob.glob(os.path.join(this_dir,'camera_numpy.zip'))[0]
    assert os.path.exists(return_dict['camera_numpy_zip'])
    return_dict['camera_size_MB'] = os.path.getsize(return_dict['camera_numpy_zip'])/1000/1000
    return return_dict

# =============================================================================
# Utility: Process frames
# =============================================================================
def process_jpg_zip(this_dir):
    return_dict = dict()
    
    if os.path.exists(os.path.join(this_dir,'jpg_images.zip')):
        return_dict['jpg_zip'] = glob.glob(os.path.join(this_dir,'jpg_images.zip'))[0]
        return_dict['jpg_zip_size_MB'] = os.path.getsize(return_dict['jpg_zip'])/1000/1000
        
        # Look inside
        with zipfile.ZipFile(return_dict['jpg_zip'], "r") as f:
            #fnames = (os.path.splitext(name) for name in f.namelist())
            return_dict['num_jpgs'] = len(f.namelist())
    else:
        raise
    return return_dict

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
        img = arrays[k]
        arrays[k]
        out_path = os.path.join(path_jpg,'{}.jpg'.format(k))
        cv2.imwrite(out_path, img)
    logging.debug("Wrote {} .jpg to {}".format(len(timestamps),path_jpg))
    return path_jpg

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

# =============================================================================
# Process json
# =============================================================================
def process_time_steps(camera_zip_path, json_zip_path):
    return_dict = dict()
    # Get the record timestamps
    numpy_timestamps = check_numpy(camera_zip_path)
    json_timestamps = check_json(json_zip_path)
    
    # Ensure timestamp alignment
    assert numpy_timestamps == json_timestamps, "Temporal alignment failure"
    
    # Analysis of timesteps
    timestamps = pd.Series(numpy_timestamps)
    
    return_dict['num_records'] =  len(timestamps)
    
    return_dict['elapsed_time'] = timestamps.iloc[-1] - timestamps.iloc[0]
    return_dict['elapsed_time_mins'] = return_dict['elapsed_time'].total_seconds() / 60
    
    # Analysis of delta-times
    ts_deltas = (timestamps-timestamps.shift()).fillna(0)
    stats = ts_deltas[0:-1].describe()
    
    return_dict['ts_deltas_mean'] = stats['mean'].total_seconds() * 1000
    return_dict['ts_deltas_std'] = stats['std'].total_seconds() * 1000
    return return_dict

def process_datetime(index_timestamp):
    return_dict = dict()
    return_dict['this_dt'] = datetime.datetime.strptime(index_timestamp, '%Y%m%d %H%M%S')
    return_dict['this_dt_iso'] = return_dict['this_dt'].isoformat()
    return_dict['this_dt_nice'] = return_dict['this_dt'].strftime("%A %d %b %H:%M")
    return return_dict


def process_json_records(this_dir):
    this_return_dict = dict()
    this_return_dict['json_record_zip'] = glob.glob(os.path.join(this_dir,'json_records.zip'))[0]
    this_return_dict['json_size_MB'] = os.path.getsize(this_return_dict['json_record_zip'])/1000/1000
    #dataset_def['num_json'] = os.path.getsize(dataset_def['json_record_zip'])/1000/1000
    
    
    this_return_dict['df_record'] = os.path.join(this_dir,'df_record.pck')
    
    if not os.path.exists(this_return_dict['df_record']):
        create_record_df(this_return_dict['json_record_zip'],this_return_dict['df_record'])
        assert os.path.exists(this_return_dict['df_record'])
    
    return this_return_dict


def create_record_df(json_zip,out_path):
    json_records = list()
    with zipfile.ZipFile(json_zip, "r") as f:
        json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
        
        for json_file in json_file_paths:
            this_fname = os.path.splitext(json_file)[0] 
            this_timestep = this_fname.split('_')[1]
            d = f.read(json_file)
            d = json.loads(d.decode("utf-8"))
            d['timestamp'] = this_timestep
            json_records.append(d)
    logging.debug("Returning {} json records from {}".format(len(json_file_paths),json_zip))
    df_records = pd.DataFrame(json_records)  
    
    df_records.to_pickle(out_path)
    logging.debug("Saved records to {}".format(out_path))



    
#%% PROCESS ALL DATASETS!

#this_data_dir = LOCAL_PROJECT_PATH

#df_datasets = df_checkfiles
def get_datasets(df_datasets):
    
    """
    Iterate over each dataset.
    For each, create a new dictionary containing information. 
    Create any new files that are missing. 
    
    """
    
    # Iterate over each dataset
    dataset_def_list = list()
    for i,this_ds in df_datasets.iterrows():
        logging.debug("***********************".format())
        logging.debug("Dataset: {}".format(i))
        
        
        
        dataset_def = dict()
        #dataset_def['this_dir'] = this_dir
        #print(i,this_ds)
        #continue
        #return
        #for i,this_ds in enumerate(df_datasets['this_dir']):
        
        #    return
        # Date time from directory name
        dataset_def.update(process_datetime(i))
        
        # Numpy camera arrays
        dataset_def.update(check_camera_zip(this_ds['this_dir']))
        
        # json state records
        dataset_def.update(process_json_records(this_ds['this_dir']))
            
        # JPG zip
        try:
            dataset_def.update(process_jpg_zip(this_ds['this_dir']))
        except:
            pass
        
        # Video
        
        # If the JPG zip doesn't exist, create it
        # THIS IS CURRENTLY DISABLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not this_ds['jpg_images.zip'] and False:
            # JPG images
            #dataset_def = create_jpgs(dataset_def)
    
            # Write the JPG's
            #if not dataset_def['num_jpgs']:
            raise Exception("UPDATE THIS")
            path_jpg = write_jpg(dataset_def)
                # Reload the jpg definition!
             #   dataset_def = process_jpgs(dataset_def)
            
            # Zip them
            path_zip = os.path.join(dataset_def['this_dir'],'jpg_images.zip')
            zip_jpgs(path_jpg,path_zip)
            delete_jpgs(path_jpg)
            
            #dataset_def = zip_jpgs(dataset_def)

            dataset_def = process_jpg_zip(dataset_def)
            
        # Time step analysis
        dataset_def.update(process_time_steps(dataset_def['camera_numpy_zip'], dataset_def['json_record_zip']))
        

        #assert dataset_def['flg_jpg_zip_exists'], "jpg zip does not exist, {}".format(dataset_def)
        
        assert dataset_def['num_jpgs'] == dataset_def['num_records']
        
        logging.debug("Dataset {}: recorded on {}".format(i, dataset_def['this_dt_nice']))
        
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
        this_series = pd.Series(dataset_def)
        this_series.name  = i
        dataset_def_list.append(this_series)
    
    # Done
    this_df_datasets = pd.DataFrame(dataset_def_list)
    #this_df_datasets = this_df_datasets.sort_values(['this_dt']).reset_index(drop=True)        
    
    return  this_df_datasets

df_datasets_processed = get_datasets(df_checkfiles)
df_all_datasets = df_checkfiles.join(df_datasets_processed)

#%%

print(tabulate(df_all_datasets.loc[:,['camera_size_MB','elapsed_time']],headers="keys"))

for c in df_all_datasets.columns:
    print(c)
#%%
def select_data(this_df):
    head_str = "{:<20}  {:<30} {:<12} {:<15} {:>30}"
    row_str =  "{:<20}  {:<30} {:<12} {:>15.0f} {:>20.1f}"
    fields = ('this_dt_nice','num_records','camera_size_MB', 'elapsed_time_mins')
    head = ['idx'] + [*fields]
    print(head_str.format(*head))
    for i,r in this_df.iterrows():
        this_row_str = [r[fld] for fld in fields]
        print(row_str.format(i,*this_row_str))
    
    #ds_idx = int(input("Select dataset number:") )
    #this_dataset = this_df.iloc[ds_idx]
    #return this_dataset
selected_data = select_data(df_all_datasets)

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