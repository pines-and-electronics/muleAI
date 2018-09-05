
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

#%%
#these_records = these_records[0:100]

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
#%% Paths

LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)

# SALIENCY IMAGES
# This is where the raw saliency (black/white) jpg frames are written
JPG_OUT_DIR = 'frames_saliency'
path_saliency_frames = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,JPG_OUT_DIR)
if not os.path.exists(path_saliency_frames): 
    os.makedirs(path_saliency_frames)

# BLEND IMAGES
# This is where the mask is overlayed on the driving frames
SALIENCY_BLEND_DIR = 'frames_saliency_blend'
path_saliency_blend_frames = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,SALIENCY_BLEND_DIR)
if not os.path.exists(path_saliency_blend_frames): 
    os.makedirs(path_saliency_blend_frames)

# Frames dir
RAW_FRAME_DIR = 'jpg_images'
path_raw_frames = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,RAW_FRAME_DIR)
if not os.path.exists(path_raw_frames): 
    os.makedirs(path_raw_frames)
    
#%% Load the dataset
#THIS_DATASET = '20180807 194733'
#THIS_DATASET = '20180807 194733'
THIS_DATASET = "20180904 192907"
data1 = AIDataSet(LOCAL_PROJECT_PATH,THIS_DATASET)

#%% Load the predictions and model, augment the dataset with these predictions
THIS_MODEL_ID = 'model 20180904 225308'
MODEL_VERSION = 'weights Loss 0.69 Epoch 06.h5'
path_model = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,THIS_MODEL_ID,MODEL_VERSION)
assert os.path.exists(path_model)
this_model = ks.models.load_model(path_model)
logging.debug("Loaded model '{}' from {}".format(MODEL_VERSION,THIS_MODEL_ID))

#%% Augment with predictions


#%% Load
df_records = pd.read_pickle(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',THIS_MODEL_ID+' predicted.pck'))
frames_npz=np.load(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'))
df_records.index = df_records['timestamp']
df_records.sort_index(inplace=True)
logging.debug("Loaded {} frames from {}".format(len(frames_npz),THIS_DATASET))
logging.debug("Loaded {} records from {}".format(len(df_records),THIS_DATASET))
#this_model
path_model = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',THIS_MODEL_ID+' model.h5' )
this_model = ks.models.load_model(path_model)


#%%
class DataSet():
    def __init__(self,path_dataset):
        self.path_dataset = path_dataset
        logging.debug("Dataset located at {}".format(self.path_dataset))

    def load_records(self,path_records):
        self.df_records = pd.read_pickle(path_records)
        logging.debug("Loaded {} records from {}".format(len(self.df_records),path_records))
        self.df_records.index = self.df_records['timestamp']
        self.df_records.sort_index(inplace=True)
        logging.debug("Sorted records".format())    

    def load_frames_npz(self,path_frames_npz):
        
        self.frames_npz = np.load(path_frames_npz)
        assert len(self.df_records) == len(self.frames_npz)
        self.frames = np.stack([self.frames_npz[idx] for idx in self.df_records.index], axis=0)
        
        logging.debug("Loaded {} frames from {}".format(len(self.frames),THIS_DATASET))

    def sample_n(self,n):
        #self.df_records.sample(n)
        these_idx = ds.df_records.sample(3).index
        these_locs = [ds.df_records.index.get_loc(idx) for idx in these_idx]
        
        return self.frames[these_locs], self.df_records.loc[these_idx,:]

    def get_full_records(self, this_indices):
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
        this_frames, this_df_records, 
        
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

ds = DataSet(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET))
ds.load_records(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'))
ds.load_frames_npz(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'))
frames, recs = ds.sample_n(3)
#self.df_records






#%% 

these_records = get_full_records(frames_npz,df_records,df_records.index)


#%% Get a pure convolutional model, no dropout or other layers

img_in = ks.layers.Input(shape=(120, 160, 3), name='img_in')
x = img_in
x = ks.layers.Convolution2D(24, (5,5), strides=(2,2), activation='relu', name='conv1')(x)
x = ks.layers.Convolution2D(32, (5,5), strides=(2,2), activation='relu', name='conv2')(x)
x = ks.layers.Convolution2D(64, (5,5), strides=(2,2), activation='relu', name='conv3')(x)
x = ks.layers.Convolution2D(64, (3,3), strides=(2,2), activation='relu', name='conv4')(x)
conv_5 = ks.layers.Convolution2D(64, (3,3), strides=(1,1), activation='relu', name='conv5')(x)
convolution_part = ks.models.Model(inputs=[img_in], outputs=[conv_5])


#%%
# Get each layer of the pure Conv model
# Assign the weights from the trained model
#layer_num = '3'
for layer_num in ('1', '2', '3', '4', '5'):
    this_pureconv_layer = convolution_part.get_layer('conv' + layer_num)
    these_weights = this_model.get_layer('conv2d_' + layer_num).get_weights()
    this_pureconv_layer.set_weights(these_weights)
    #convolution_part.get_layer('conv' + layer_num).set_weights(model.get_layer('conv' + layer_num).get_weights())
#blmodel.layers
#blmodel.summary()
#r = convolution_part.get_layer('conv' + layer_num).get_weights()
#type(r)
#len(r)
logging.debug("Assigned trained model weights to convolutional layers".format())


#%%

inp = convolution_part.input                                           # input placeholder
outputs = [layer.output for layer in convolution_part.layers]          # all layer outputs
functor = ks.backend.function([inp], outputs)

logging.debug("Created backend function from weighted convolutional layers".format())


#%% Recreate the kernels and strides for each layer

kernel_3x3 = tf.constant(np.array([
        [[[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]]]
]), tf.float32)

kernel_5x5 = tf.constant(np.array([
        [[[1]], [[1]], [[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]], [[1]], [[1]]], 
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]]
]), tf.float32)

layers_kernels = {5: kernel_3x3, 4: kernel_3x3, 3: kernel_5x5, 2: kernel_5x5, 1: kernel_5x5}

layers_strides = {5: [1, 1, 1, 1], 4: [1, 2, 2, 1], 3: [1, 2, 2, 1], 2: [1, 2, 2, 1], 1: [1, 2, 2, 1]}

#%%
#img = one_img_array
def compute_visualisation_mask(img,functor,layers_kernels_dict,layers_strides_dict):
    activations = functor([np.array([img])])
    #The upscaled activation changes each loop (layer)
    upscaled_activation = np.ones((3, 6))
    layer = 5
    for layer in [5, 4, 3, 2, 1]:
        averaged_activation = np.mean(activations[layer], axis=3).squeeze(axis=0) * upscaled_activation
        output_shape = (activations[layer - 1].shape[1], activations[layer - 1].shape[2])
        x = tf.constant(
            np.reshape(averaged_activation, (1,averaged_activation.shape[0],averaged_activation.shape[1],1)),
            tf.float32
        )
        conv = tf.nn.conv2d_transpose(
            x, layers_kernels_dict[layer],
            output_shape=(1,output_shape[0],output_shape[1], 1), 
            strides=layers_strides_dict[layer], 
            padding='VALID'
        )
        with tf.Session() as session:
            result = session.run(conv)
        upscaled_activation = np.reshape(result, output_shape)
        
    final_visualisation_mask = upscaled_activation
    return (final_visualisation_mask - np.min(final_visualisation_mask))/(np.max(final_visualisation_mask) - np.min(final_visualisation_mask))


#TESTING! 
result = compute_visualisation_mask(these_records[0]['frame'],functor,layers_kernels,layers_strides)
plt.imshow(result)

#%%
#plt.imshow(result)
output_interval = 100
start = time.time()
cnt = 0
for rec in these_records:
    #print(rec['timestamp'])
    path_out = os.path.join(path_saliency_frames,rec['timestamp_raw']+'.png')
    #path_out = os.path.join(path_saliency_frames,'1533132117808.png')
    
    if os.path.exists(path_out):
        #print("Skip",path_out)
        continue
    cnt += 1
    salient_mask = compute_visualisation_mask(rec['frame'],functor,layers_kernels,layers_strides)
    # Stack it 3 times to get into RGB dimension
    salient_mask_stacked = np.dstack((salient_mask,salient_mask,salient_mask))
    plt.imsave(path_out, salient_mask_stacked)
    
    if cnt%output_interval == 0:
        duration = time.time()  - start
        print("Last {} frames took {:.1f}s".format(output_interval,duration))
        start = time.time()
        #print("{:>4} of {} {}".format(counter, total_images, os.path.split(path)[1]), end = "")
        #print(", {:.1f}s, ".format(duration), end="")
        #print("{:.1f} min remaining".format(duration/60*(cnt_remaining-1)))


#%%
#this_ts = these_records[0]['timestamp_raw']

#raw_frame = these_records[0]['frame']
#npz_object = frames_npz
#path_img_msk_pngs = path_saliency_frames 
#rec = these_records[0]
def blend_frame_and_save(npz_object,path_img_msk_pngs,rec):
    BLUR_SIZE = 8
    #BLUR_FACTOR = 1/25
    BLUR_FACTOR = 1/10
    
    #print(rec['timestamp_raw'])
    raw_frame = npz_object[rec['timestamp_raw']]
    
    #raw_frame.shape
    #raw_frame.dtype
    # Raw mask
    path_this_salient_frame = os.path.join(path_saliency_frames,rec['timestamp_raw']+'.png')
    
    saliency_frame = plt.imread(path_this_salient_frame)[:,:,:3]
    #saliency_frame = plt.imread(path_this_salient_frame)
    #saliency_frame *= 255
    #saliency_frame = saliency_frame.astype(np.uint8)
    #saliency_frame.shape
    #saliency_frame.dtype

    # Apply RGB transform
    saliency_frame.setflags(write=1)
    saliency_frame[:,:,0] = saliency_frame[:,:,0] * 10
    saliency_frame[:,:,1] = saliency_frame[:,:,1] * 0.9
    saliency_frame[:,:,2] = saliency_frame[:,:,2] * 0
    

    #plt.imshow(saliency_frame)  


    
    
    
    blur_kernel = np.ones((BLUR_SIZE,BLUR_SIZE),np.float32) * BLUR_FACTOR
    saliency_frame_blurred = cv2.filter2D(saliency_frame,-1,blur_kernel)
    #plt.imshow(saliency_frame_blurred)
    
    #saliency_frame_blurred - cv2.cvtColor(saliency_frame_blurred, cv2.CV_8UC1);
    
    #saliency_frame_blurred = cv2.applyColorMap(saliency_frame_blurred, cv2.COLORMAP_JET)

    alpha = 0.004
    beta = 1.0 - alpha
    #beta = 1
    gamma = 0.0
    
    #factor = 10
    
    blend = cv2.addWeighted(raw_frame.astype(np.float32), alpha, saliency_frame_blurred, beta, gamma)

    #plt.imshow(blend)
    
    #plt.imshow(raw_frame)
    #plt.imshow(saliency_frame_masked, alpha=0)
    
    
    if 0:
        mask = np.ma.masked_where(saliency_frame_blurred>0, saliency_frame_blurred)
        saliency_frame_masked = np.ma.masked_array(saliency_frame_blurred,mask)
        plt.imshow(saliency_frame_masked)
        plt.imshow(mask)
        
        gray_image = cv2.cvtColor(saliency_frame_blurred, cv2.COLOR_BGR2GRAY)
        backtorgb = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)    
        plt.imshow(backtorgb)    
        
        subtracted_raw = raw_frame - backtorgb
        #plt.imshow(raw_frame - backtorgb)
        
        dst = cv2.add(subtracted_raw,saliency_frame_blurred)
        plt.imshow(dst)    

    
    #Image.
    
    #saliency_frame = np.asarray(Image.open(path_this_salient_frame))
    #im = Image.open(path_this_salient_frame)

    #saliency_frame.dtype
    #saliency_image = Image.open(path_this_salient_frame)
    
    #saliency_image_rgb = saliency_image.convert('RGB')
    #rgb_image_np = np.asarray(saliency_image_rgb)
    #plt.imshow(rgb_image_np)
    #plt.imshow(saliency_frame)
    #saliency_frame = rgb_image_np
    #np.max(saliency_frame[:,:,0])
    #np.max(saliency_frame[:,:,1])
    #np.max(saliency_frame[:,:,2])
    

    
    # Blurred mask
    

    if 0:
        img2gray = cv2.cvtColor(saliency_frame_blurred,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0.01, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        
        background_image = cv2.bitwise_and(raw_frame,raw_frame,mask = mask_inv)
        new = raw_frame.paste(img, offset)
    
        
        plt.imshow(img2gray)
        plt.imshow(mask)
        plt.imshow(mask_inv)
        plt.imshow(background_image)    
    
        #img2_fg = cv2.bitwise_and(saliency_frame_blurred,saliency_frame_blurred,mask = mask)
        # Put logo in ROI and modify the main image
        
        plt.imshow(raw_frame)
        dst = cv2.add(raw_frame,saliency_frame_blurred)
        plt.imshow(dst)
        
        
        img1[0:rows, 0:cols ] = dst
    
    
    
    #raise
    #saliency_frame_blurred[:,:,1] = saliency_frame_blurred[:,:,0] * 2
    #saliency_frame_blurred[:,:,1] = saliency_frame_blurred[:,:,1] * 1
    #saliency_frame_blurred[:,:,2] = saliency_frame_blurred[:,:,2] * 0.
    
    # Color mask
    #saliency_frame_blurred_colored = cv2.cvtColor(saliency_frame,cv2.COLOR_BGR2LAB)
    #saliency_frame_blurred_colored = cv2.cvtColor(saliency_frame_blurred,cv2.COLOR_RGB2HSV)
    #plt.imshow(saliency_frame_blurred_colored)
    
    #plt.imshow(raw_frame)
    

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
