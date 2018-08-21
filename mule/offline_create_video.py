
# Requirements:
# 3042  conda create --name ffmpeg-video
# 3043  act ffmpeg-vido
# 3044  act ffmpeg-video
# 3045  conda install matplotlib
# 3046  conda install pandas
# 3047  conda install cv2
# 3048  conda install -c menpo opencv
# 3049  conda install -c conda-forge ffmpeg
# DOESN'T WORK!!
# conda remove opencv
##pip install opencv-python

#%%
import os

#import tensorflow as tf
#import numpy as np

import cv2
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
mpl.__version__
#import matplotlib
#from matplotlib.animation import FuncAnimation

#assert 'ffmpeg' in matplotlib.animation.writers.list(), "Need to install ffmpeg!"
import datetime
#ffmpeg_writer = matplotlib.animation.FFMpegWriter

#%matplotlib inline
#
import sys
import zipfile

import logging

#import matplotlib.pyplot as plt
#from matplotlib import animation
#from IPython.display import display, HTML
import glob

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

#path_frame_zip = r"/home/batman/MULE DATA/20180801 192613/jpg_images.zip"
#path_video_out = r"/home/batman/MULE DATA/20180801 192613/this_vid.mp4"



#%%

def get_frames(path_frame_zip):
    # Look inside
    frames = list()
    with zipfile.ZipFile(path_frame_zip, "r") as this_zip:
        #fnames = [os.path.splitext(name) for name in this_zip.namelist()]
        for this_jpg in this_zip.namelist():
            this_frame_dict = dict()
            #print(this_jpg)
            timestamp = os.path.splitext(this_jpg)[0]
            datetime_stamp = datetime.datetime.fromtimestamp(int(timestamp)/1000)
            datetime_str = datetime_stamp.isoformat()
            img_bytes = this_zip.read(this_jpg)
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)
            
            #frames[datetime_str] = img
            this_frame_dict['array'] = img
            this_frame_dict['datetime_stamp'] = datetime_stamp
            this_frame_dict['datetime_str'] = datetime_str
            frames.append(this_frame_dict)

    # Sort the frames!
    frames = sorted(frames, key=lambda k: k['datetime_stamp']) 
    logging.debug("Collected {} frames".format(len(frames)))

    return frames

#%%
def write_video(frames,path_video_out):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(path_video_out, fourcc, 20, (160,120))
    
    for this_frame_dict in frames:
        
        #image_path = os.path.join(dir_path, image)
        frame = this_frame_dict['array']
    
        out.write(frame) # Write out frame to video
    
        #cv2.imshow('video',frame)
        #if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        #    break
    logging.debug("Wrote {} frames to {}".format(len(frames),path_video_out))
    
    out.release()


#%%
        
#this_data_dir = LOCAL_PROJECT_PATH
def write_videos(this_data_dir):
    directoy_list = glob.glob(os.path.join(this_data_dir,'*'))
    
    # Iterate over each directory
    for i,this_dir in enumerate(directoy_list):
        logging.debug("Dataset {} {}".format(i, this_dir))
        
        dataset_def = dict()
        dataset_def['this_dir'] = this_dir
        dataset_def['path_video_out'] = os.path.join(dataset_def['this_dir'],'video.mp4')
        print(os.path.exists(dataset_def['path_video_out']))
        if not os.path.exists(dataset_def['path_video_out']):
            # JPG zip
            dataset_def = process_jpg_zip(dataset_def)
            
            frames = get_frames(dataset_def['jpg_zip'])
            
            write_video(frames,dataset_def['path_video_out'])
#%% 
        
write_videos(LOCAL_PROJECT_PATH)

        
#%%
#    
#    print(fnames)
#    img.
#    
#    
#    
#        timestamps, extensions = zip(*fnames)
#        assert all(ext == '.npy' for ext in extensions)
#        #[ts.isoformat() for ts in timestamps]
#        
#        #ts = timestamps[0]
#        datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
#        datetime_stamps.sort()
#        return datetime_stamps

#%% 

## Images
##path_root = r"/home/batman/d2/data/tub_67"
#path_frame_zip = r"/home/batman/MULE DATA/20180801 192613/jpg_images.zip"
#
#
#
#path_images_dir = os.path.join(path_root,'imgs')
#assert os.path.exists(path_images_dir)
#glob_string = path_images_dir + '/*.jpg'
#print(glob_string)
#all_images = sorted(iglob(glob_string))
#print(len(all_images), 'Images found in', path_images_dir)
#
## Weights
#path_weights = os.path.join(path_root,'mypilot73.h5')
#assert os.path.exists(path_weights), path_weights
#print('Weight file:',path_weights)