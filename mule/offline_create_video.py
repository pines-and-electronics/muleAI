
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
#import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
mpl.__version__
#import matplotlib
#from matplotlib.animation import FuncAnimation

#assert 'ffmpeg' in matplotlib.animation.writers.list(), "Need to install ffmpeg!"
import datetime
ffmpeg_writer = matplotlib.animation.FFMpegWriter

#%matplotlib inline
#
import zipfile

import matplotlib.pyplot as plt
#from matplotlib import animation
#from IPython.display import display, HTML

#import time
#%%

path_frame_zip = r"/home/batman/MULE DATA/20180801 192613/jpg_images.zip"

#%%

def get_frames(path_frame_zip):
    # Look inside
    frames = dict()
    with zipfile.ZipFile(path_frame_zip, "r") as this_zip:
        fnames = [os.path.splitext(name) for name in this_zip.namelist()]
        for this_jpg in f.namelist():
            #print(this_jpg)
            timestamp = os.path.splitext(this_jpg)[0]
            datetime_stamp = datetime.datetime.fromtimestamp(int(timestamp)/1000)
            datetime_str = datetime_stamp.isoformat()
            img_bytes = this_zip.read(this_jpg)
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)
            frames[datetime_str] = img
    return frames

frames = get_frames(path_frame_zip)


#%%




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

# Images
#path_root = r"/home/batman/d2/data/tub_67"
path_frame_zip = r"/home/batman/MULE DATA/20180801 192613/jpg_images.zip"



path_images_dir = os.path.join(path_root,'imgs')
assert os.path.exists(path_images_dir)
glob_string = path_images_dir + '/*.jpg'
print(glob_string)
all_images = sorted(iglob(glob_string))
print(len(all_images), 'Images found in', path_images_dir)

# Weights
path_weights = os.path.join(path_root,'mypilot73.h5')
assert os.path.exists(path_weights), path_weights
print('Weight file:',path_weights)