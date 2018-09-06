
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

#%%
#these_records = these_records[0:100]



class SaliencyGen():
    """Aggregates the ModelledDataSet class, and operates to produce frames
    """
    def __init__(self,modelled_dataset):
        # Get the model
        self.modelled_dataset = modelled_dataset
        assert modelled_dataset.has_predictions
        self.model_folder = modelled_dataset.model_folder
        self.path_model_dir = modelled_dataset.path_model_dir
        logging.debug("Saliency gen for model ID: {}".format(self.model_folder))
        logging.debug("Loaded model accuracy: {:0.1f}%".format(self.modelled_dataset.raw_accuracy*100))
        
        # Original raw images
        self.path_jpgs_dir = modelled_dataset.path_jpgs_dir
        logging.debug("Source orginal raw images are in folder: {}".format(self.path_jpgs_dir))
        jpg_files = glob.glob(os.path.join(self.path_jpgs_dir,'*.jpg'))
        logging.debug("{} jpgs found".format(len(jpg_files)))
        
        # New saliency mask jpgs
        self.path_saliency_jpgs = os.path.join(self.path_model_dir,'saliency_mask_jpgs')
        if not os.path.exists(self.path_saliency_jpgs): 
            os.makedirs(self.path_saliency_jpgs)
        logging.debug("Saliency JPG output folder: {}".format(self.path_saliency_jpgs))
        
        # Boosted saliency mask jpgs
        self.path_boosted_saliency_jpgs = os.path.join(self.path_model_dir,'boosted_saliency_mask_jpgs')
        if not os.path.exists(self.path_boosted_saliency_jpgs): 
            os.makedirs(self.path_boosted_saliency_jpgs)
        logging.debug("BOOSTED Saliency JPG output folder: {}".format(self.path_boosted_saliency_jpgs))
        
        # New saliency frames
        self.path_frames_jpgs = os.path.join(self.path_model_dir,'saliency_frames_jpgs')
        if not os.path.exists(self.path_frames_jpgs): 
            os.makedirs(self.path_frames_jpgs)        
        logging.debug("Combined HUD frames output folder: {}".format(self.path_frames_jpgs))
    
    def gen_pure_CNN(self):
        # Get a pure convolutional model, no dropout or other layers
        img_in = ks.layers.Input(shape=(120, 160, 3), name='img_in')
        x = img_in
        x = ks.layers.Convolution2D(24, (5,5), strides=(2,2), activation='relu', name='conv1')(x)
        x = ks.layers.Convolution2D(32, (5,5), strides=(2,2), activation='relu', name='conv2')(x)
        x = ks.layers.Convolution2D(64, (5,5), strides=(2,2), activation='relu', name='conv3')(x)
        x = ks.layers.Convolution2D(64, (3,3), strides=(2,2), activation='relu', name='conv4')(x)
        conv_5 = ks.layers.Convolution2D(64, (3,3), strides=(1,1), activation='relu', name='conv5')(x)
        convolution_part = ks.models.Model(inputs=[img_in], outputs=[conv_5])
        self.convolutional_model = convolution_part
        logging.debug("Generated a pure CNN {}".format(self.convolutional_model))

    def get_layers(self):
        # Get each layer of the pure Conv model
        # Assign the weights from the trained model
        logging.debug("Retreiving and copying weights from the model, to the pure CNN".format())
        for layer_num in ('1', '2', '3', '4', '5'):
            
            this_pureconv_layer = self.convolutional_model.get_layer('conv' + layer_num)
            this_layer_name = 'conv2d_' + layer_num
            #print("Copied weights from loaded", this_layer_name, "to the pure CNN")
            these_weights = self.modelled_dataset.model.get_layer(this_layer_name).get_weights()
            this_pureconv_layer.set_weights(these_weights)
        logging.debug("Assigned trained model weights to all convolutional layers".format())


    def saliency_tf_function(self):
        inp = self.convolutional_model.input                                           # input placeholder
        outputs = [layer.output for layer in self.convolutional_model.layers]          # all layer outputs
        saliency_function = ks.backend.function([inp], outputs)
        self.saliency_function = saliency_function
        logging.debug("Created tensorflow pipeliine (saliency_function) from weighted convolutional layers".format())

    def get_kernels(self):
        """ Recreate the kernels and strides for each layer
        """
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
        
        self.layers_kernels = {5: kernel_3x3, 4: kernel_3x3, 3: kernel_5x5, 2: kernel_5x5, 1: kernel_5x5}
        
        self.layers_strides = {5: [1, 1, 1, 1], 4: [1, 2, 2, 1], 3: [1, 2, 2, 1], 2: [1, 2, 2, 1], 1: [1, 2, 2, 1]}
            
        logging.debug("Assigned layers_kernels and layers_strides".format())
        
    def write_saliency_mask_jpgs(self,number=None):
        frames_npz = np.load(self.modelled_dataset.path_frames_npz)
        if not number:
            number = len(self.modelled_dataset.df)
            
        with LoggerCritical(), NoPlots():
            for idx in tqdm.tqdm(self.modelled_dataset.df.index[0:number]):
                rec = self.modelled_dataset.df.loc[idx]
                path_out = os.path.join(self.path_saliency_jpgs,rec['timestamp']+'.png')
    
                #print(idx,rec)
                
                # Get a frame array, and shape it to 4D
                img_array = frames_npz[idx]
                img_array = np.expand_dims(img_array, axis=0)
                activations = self.saliency_function([img_array])
                
                # The upscaled activation changes each loop (layer)
                upscaled_activation = np.ones((3, 6))
                for layer in [5, 4, 3, 2, 1]:
                    averaged_activation = np.mean(activations[layer], axis=3).squeeze(axis=0) * upscaled_activation
                    output_shape = (activations[layer - 1].shape[1], activations[layer - 1].shape[2])
                    x = tf.constant(
                        np.reshape(averaged_activation, (1,averaged_activation.shape[0],averaged_activation.shape[1],1)),
                        tf.float32
                    )
                    conv = tf.nn.conv2d_transpose(
                        x, self.layers_kernels[layer],
                        output_shape=(1,output_shape[0],output_shape[1], 1), 
                        strides=self.layers_strides[layer], 
                        padding='VALID'
                    )
                    with tf.Session() as session:
                        result = session.run(conv)
                    upscaled_activation = np.reshape(result, output_shape)
                    
                    salient_mask = (upscaled_activation - np.min(upscaled_activation))/(np.max(upscaled_activation) - np.min(upscaled_activation))
                    
                    # Make an RGB 3-channel image            
                    salient_mask_stacked = np.dstack((salient_mask,salient_mask,salient_mask))
                    
                    # Save it to JPG
                    plt.imsave(path_out, salient_mask_stacked)
                
    def blend_simple(self,blur_rad,strength,num_frames = None):
        #
        logging.debug("blur_rad {}, strength {}".format(blur_rad,strength))
        
        source_folder = os.path.split(self.path_saliency_jpgs)[1]
        target_folder = os.path.split(self.path_boosted_saliency_jpgs)[1]
        jpg_files = glob.glob(os.path.join(self.path_saliency_jpgs,'*.png'))
        logging.debug("Boosting {} frames at {} to {}".format(len(jpg_files),source_folder,target_folder))
        
        frames_npz = np.load(self.modelled_dataset.path_frames_npz)
        
        # For testing, write a sample
        if not num_frames:
            num_frames = len(jpg_files)
            
        with LoggerCritical(), NoPlots():
            for img_path in tqdm.tqdm(jpg_files[0:num_frames]):
                #print(img_path)
                _,fname = os.path.split(img_path)
                timestamp,_ = os.path.splitext(fname)
                #print(timestamp)
                saliency_frame = plt.imread(img_path)[:,:,:3]
                raw_frame = frames_npz[timestamp]
                
                if 0: # Try adjusting brightness and contrast...
                    b = 0. # brightness
                    c = 64.  # contrast
                
                    #call addWeighted function, which performs:
                    #    dst = src1*alpha + src2*beta + gamma
                    # we use beta = 0 to effectively only operate on src1
                    saliency_frame = cv2.addWeighted(saliency_frame, 1. + c/127., saliency_frame, 0, b-c)                                
                
                saliency_frame.setflags(write=1)
                saliency_frame[:,:,0] = saliency_frame[:,:,0] * 0.5
                saliency_frame[:,:,1] = saliency_frame[:,:,1] * 1.2
                saliency_frame[:,:,2] = saliency_frame[:,:,2] * 0
                blur_kernel = np.ones((blur_rad,blur_rad),np.float32) * strength
                saliency_frame_blurred = cv2.filter2D(saliency_frame,-1,blur_kernel)


                #saliency_frame_blurred = cv2.GaussianBlur(saliency_frame,(1,1),0)
                
                alpha = 0.004
                beta = 1.0 - alpha
                gamma = 0.0
                
                blend = cv2.addWeighted(raw_frame.astype(np.float32), alpha, saliency_frame_blurred, beta, gamma)
                #plt.imshow(blend)
                
                
                path_out = os.path.join(self.path_boosted_saliency_jpgs,fname)
                plt.imsave(path_out, blend)

        # Raw masks
        pass
    
    def create_HUD_frames(self):
        source_folder_jpg = os.path.split(self.path_saliency_jpgs)[1]
        target_folder = os.path.split(self.path_frames_jpgs)[1]
        jpg_files = glob.glob(os.path.join(self.path_boosted_saliency_jpgs,'*.png'))
        logging.debug("Creating {} HUD frames from {} to {}".format(len(jpg_files),source_folder_jpg,target_folder))        

        with LoggerCritical(), NoPlots():
            for img_path in tqdm.tqdm(glob.glob(self.path_boosted_saliency_jpgs + r"/*.png")):
                print(img_path)
                _,fname = os.path.split(img_path)
                index,_ = os.path.splitext(fname)
                pathpart_source_imgs = self.model_folder + r"/" + 'boosted_saliency_mask_jpgs'
                frame_figure = this_saliency.modelled_dataset.gen_record_frame(index, source_jpg_folder=pathpart_source_imgs, source_ext = '.png')
                
                # Save it to jpg
                #path_jpg = os.path.join(OUT_PATH,idx + '.jpg')
                path_jpg = os.path.join(self.path_frames_jpgs, index + ".jpg")
                frame_figure.savefig(path_jpg)
        logging.debug("Wrote frames to {}".format(self.path_frames_jpgs))
        

    def blend_PIL(self,blur_rad,map_name,strength):
        """A more advanced boosting pipeline
        """
        logging.debug("blur_rad {}, map_name {}, strength {}".format(blur_rad,map_name,strength))

        pass

#%% Load the DataSet, and train the predictions
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
THIS_DATASET = "20180904 192907"
#THIS_MODEL_ID = 'model 20180906 154310'
THIS_MODEL_ID = 'model 20180906 165918'
this_modelled_ds = ModelledDataSet(LOCAL_PROJECT_PATH,THIS_DATASET,THIS_MODEL_ID)
this_modelled_ds.load_best_model()
this_modelled_ds.model.summary()
this_modelled_ds.make_predictions()

#%% Generate the saliency frames
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
