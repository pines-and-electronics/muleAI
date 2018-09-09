
#%%
#assert 'ffmpeg' in mpl.animation.writers.list(), "Need to install ffmpeg!"

#%%
#import os
#
#from keras.layers import Input, Dense, merge
#from keras.models import Model
#from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
#from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K
#
#from glob import iglob
#
#import tensorflow as tf
#import numpy as np
#
#import cv2
#import numpy as np
#
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib as mpl
#assert 'ffmpeg' in mpl.animation.writers.list(), "Need to install ffmpeg!"
#
#%matplotlib inline
#
#import matplotlib.pyplot as plt
#from matplotlib import animation
#from IPython.display import display, HTML
#
#import time
import os
import tensorflow as tf
import glob
import datetime
from tensorflow.python import keras as ks
import cv2
import time
from PIL import Image
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
    
class NoPlots:
    def __enter__(self):
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.ioff()
    def __exit__(self, type, value, traceback):
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.ion()    

#%% 1) Load the DataSet, and train the predictions AGAIN
        
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
THIS_DATASET = "20180904 192907"
#THIS_MODEL_ID = 'model 20180906 154310'
THIS_MODEL_ID = 'model 20180906 165918'
this_modelled_ds = ModelledDataSet(LOCAL_PROJECT_PATH,THIS_DATASET,THIS_MODEL_ID)
this_modelled_ds.load_best_model()
this_modelled_ds.model.summary()
this_modelled_ds.make_predictions()

#%% 2) Generate the saliency black & white frames
this_saliency = SaliencyGen(this_modelled_ds)
this_saliency.gen_pure_CNN()
this_saliency.path_saliency_jpgs
##this_saliency.modelled_dataset.model.layers
#this_saliency.modelled_dataset.model.summary()
this_saliency.get_layers()
this_saliency.saliency_tf_function()
this_saliency.get_kernels()
if False: # This takes a while!
    this_saliency.write_saliency_mask_jpgs()
#this
#%%
raise
if False:
    this_saliency.create_HUD_frames()

#this_saliency.modelled_dataset.gen_record_frame('1536082180023', source_jpg_folder=r"model 20180906 165918/boosted_saliency_mask_jpgs", source_ext = '.png')
raise
#%%
#this_strength = 4
#this_blur = 3
this_saliency.blend_simple(2,1.5)

#%%
this_map = 'viridis'
this_map = 'seismic'
this_map = 'summer'
this_map = 'magma'
this_map = 'hot'
this_strength = 4
this_blur = 3
this_saliency.blend_PIL(this_blur,this_map,this_strength,this_map)

#%%
#this_ts = these_records[0]['timestamp_raw']

#raw_frame = these_records[0]['frame']
#npz_object = frames_npz
#path_img_msk_pngs = path_saliency_frames 
#rec = these_records[0]
def blend_frame_and_save(npz_object,path_img_msk_pngs,rec):
    BLUR_SIZE = 8
    BLUR_FACTOR = 1/10
    
    raw_frame = npz_object[rec['timestamp_raw']]

    # Raw mask
    path_this_salient_frame = os.path.join(path_saliency_frames,rec['timestamp_raw']+'.png')
    
    saliency_frame = plt.imread(path_this_salient_frame)[:,:,:3]

    # Apply RGB transform
    saliency_frame.setflags(write=1)
    saliency_frame[:,:,0] = saliency_frame[:,:,0] * 10
    saliency_frame[:,:,1] = saliency_frame[:,:,1] * 0.9
    saliency_frame[:,:,2] = saliency_frame[:,:,2] * 0
    
    blur_kernel = np.ones((BLUR_SIZE,BLUR_SIZE),np.float32) * BLUR_FACTOR
    saliency_frame_blurred = cv2.filter2D(saliency_frame,-1,blur_kernel)

    alpha = 0.004
    beta = 1.0 - alpha
    gamma = 0.0
    
    blend = cv2.addWeighted(raw_frame.astype(np.float32), alpha, saliency_frame_blurred, beta, gamma)

    return blend
    #blended.apppath_saliency_framesend(blend)
    #return blended
    #
    #return
this_blended = blend_frame_and_save(frames_npz,path_saliency_frames,these_records[0])
plt.imshow(this_blended)


this_blended = blend_frame_and_save(frames_npz,path_saliency_frames,these_records[500])
plt.imshow(this_blended)
#%% TESTING

def blend_frame_and_save2(npz_object,path_img_msk_pngs,rec,blur_rad,map_name,strength):
    #rec = these_records[500]
    path_this_salient_frame = os.path.join(path_saliency_frames,rec['timestamp_raw']+'.png')
    #sframe = plt.imread(path_this_salient_frame)
    sframe = PIL.Image.open(path_this_salient_frame)
    
    path_raw_frame = os.path.join(path_raw_frames,rec['timestamp_raw']+'.png')
    #frame = plt.imread(path_this_salient_frame)
    #frame = npz_object[rec['timestamp_raw']]
    frame = PIL.Image.open(path_raw_frame)
    red, green, blue, alpha = frame.split()
    
    #frame.show()
    #figure(figsize = (20,20))
    #plt.imshow(sframe)
    #plt.imshow(frame)
    
    
    # Blur
    blurred_image = sframe.filter(PIL.ImageFilter.GaussianBlur(radius=blur_rad))
    #plt.imshow(blurred_image)
    
    # Contrast 
    contrast = PIL.ImageEnhance.Contrast(blurred_image)
    #contrast = contrast.enhance(3)
    contrast = contrast.enhance(strength)
    
    ##plt.imshow(contrast)
    red, green, blue, alpha = contrast.split()
    raw_alpha_mask = red
    contrast.putalpha(raw_alpha_mask)
    
    # Color
    cm_hot = mpl.cm.get_cmap('inferno')
    cm_hot = mpl.cm.get_cmap('plasma')
    
    cm_hot = mpl.cm.get_cmap('jet')
    cm_hot = mpl.cm.get_cmap('magma')
    cm_hot = mpl.cm.get_cmap('ocean')
    
    #cm_hot = mpl.cm.get_cmap('copper')
    #cm_hot = mpl.cm.get_cmap('hot')
    #cm_hot = mpl.cm.get_cmap('rainbow')
    
    #cm_hot.set_under(color="black", alpha="0")
    cm_hot = mpl.cm.get_cmap('hot')
    cm_hot = mpl.cm.get_cmap('jet')
    cm_hot = mpl.cm.get_cmap(map_name)
    contrastL = contrast.convert('L')
    constrast_arr = cm_hot(np.array(contrastL))
    colored = np.uint8(constrast_arr * 255)
    colored = Image.fromarray(colored)
    #colored.putalpha(raw_alpha_mask)
    #plt.imshow(colored)
    red, green, blue, alpha = colored.split()
    #im.mode
    #im.alpha_composite()
    
    # Paste
    pasted = frame.copy()
    pasted.paste(colored,mask=raw_alpha_mask)
    #plt.imshow(pasted)
    return pasted

    if 0:
        # Subtract 
        subtracted = PIL.ImageChops.subtract(frame,contrast)
        
        # Added
        added = PIL.ImageChops.add(frame,colored)
        plt.imshow(added)
        
        
        # Composited
        composite = PIL.ImageChops.composite(frame, colored, raw_alpha_mask)
        plt.imshow(composite)
        
        
        raw_alpha_mask.alpha
        
        
        
        # BLend
        alphaBlended1 = PIL.Image.blend(frame, colored, alpha=0.5)
        plt.imshow(alphaBlended1)
        
        
        
        # Composite
        mask = blurred_image.convert('L')
        mask.mode, mask.size, 
        dir(mask)
        maskarr = np.array(mask)
        this = PIL.Image.composite(frame, im, mask)
        plt.imshow(this)
        
        
        # BLend
        alphaBlended1 = PIL.Image.blend(frame, im, alpha=.2)
        plt.imshow(alphaBlended1)
        
        #PIL.ImageColor.colormap[]
        #EDGE_ENHANCE
        #edge = PIL.ImageEnhance.EDGE_ENHANCE(blurred_image)
        #contrast = contrast.enhance(3)
        #plt.imshow(contrast,cmap='hot')
        #plt.imshow(im, 
        # Color
        #cm = plt.get_cmap('gist_rainbow')
        #im = np.array(contrast)
        #colored_image = cm(im)
        #im = Image.fromarray(colored_image)
        
        
        
        #PIL.ImageEnhance
#['viridis', 'plasma', 'inferno', 'magma']
this_map = 'viridis'

this_map = 'seismic'
this_map = 'summer'
this_map = 'magma'
this_map = 'hot'
this_strength = 1.7
this_blur = 1
this_blended1 = blend_frame_and_save2(frames_npz,path_saliency_frames,these_records[0],this_blur,this_map,this_strength)
plt.imshow(this_blended1)

this_blended2 = blend_frame_and_save2(frames_npz,path_saliency_frames,these_records[100],this_blur,this_map,this_strength)
plt.imshow(this_blended2)


this_blended3 = blend_frame_and_save2(frames_npz,path_saliency_frames,these_records[500],this_blur,this_map,this_strength)
plt.imshow(this_blended3)
 
#%% Write NPZ to PNG

for rec in these_records:
    frame = npz_object[rec['timestamp_raw']]
    #this_blended = blend_frame_and_save(frames_npz,path_saliency_frames,rec)
    path_out = os.path.join(path_raw_frames,rec['timestamp_raw']+'.png')
    plt.imsave(path_out, frame)

#%%
#%matplotlib qt
#%matplotlib inline
# Then disable output
    
    
if not os.path.exists(path_saliency_blend_frames): 
    os.makedirs(path_saliency_blend_frames)
    
plt.ioff()    
this_map = 'hot'
this_strength = 1.7
this_blur = 1
for rec in these_records:
    with LoggerCritical():   
        this_blended = blend_frame_and_save2(frames_npz,path_saliency_frames,rec,this_blur,this_map,this_strength)
        path_out = os.path.join(path_saliency_blend_frames,rec['timestamp_raw']+'.png')
        plt.imsave(path_out, this_blended)

#plt.imshow(this_blended)
#frames_npz[0]
#%%
