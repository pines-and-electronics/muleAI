#%%
def get_predictions(model, this_frames_npz, this_df_records):
    """Augment the df_records with the predictions
    """
    this_df_records['steering_pred_cats'] = pd.Series(dtype=object)
    #df_records['steering_pred_argmax'] = 
    for i,idx in enumerate(this_df_records.index):
        
        this_frame = np.expand_dims(this_frames_npz[idx],0)
        this_frame.shape
        this_pred = model.predict(this_frame)
        this_df_records.loc[idx,'steering_pred_cats'] = [this_pred]
        this_df_records.loc[idx,'steering_pred_argmax'] = np.argmax(this_pred)
        this_df_records.loc[idx,'steering_pred_signal'] = linear_unbin(this_df_records.loc[idx,'steering_pred_cats'][0])
        if i%100 == 0:
            print(i,"|", end="")
    logging.debug("Returning predictions. NB: Steering is INVERTED!!!".format())
    
    return this_df_reco


#%% Data gen OLD
class MuleDataGenerator(ks.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, path_frames, path_records, 
                 batch_size=32, dim=None, n_channels=None, n_classes=15, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path_frames = path_frames
        assert os.path.exists(self.path_frames)
        self.path_records = path_records
        assert os.path.exists(self.path_records)
        logging.debug("** Initialize datagen **".format())
        logging.debug("Frames stored at: {}".format(self.path_frames))
        logging.debug("Records stored at: {}".format(self.path_records))
        logging.debug("{} samples over batch size {} yields {} batches".format(len(list_IDs),
                                                                                   self.batch_size,
                                                                                   math.ceil(len(list_IDs)/self.batch_size),))
        
    def __len__(self):
        """Keras generator method - Denotes the number of batches per epoch
        """        
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    # GET A BATCH!
    def __getitem__(self, index): 
        """Generate one batch of data
        """         
        logging.debug("Generating batch {}".format(index))
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data by selecting these IDs
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Keras generator method - Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __get_npy_arrays(self,list_IDs_temp):
        """Custom method - get the X input arrays
        
        Open the npz file and load n frames into memory
        """
        # This is a pointer to the file
        npz_file=np.load(self.path_frames)
        #for k in list_ID_temp:
        #    npy_records.append(npz_file[k])
        #X_train = np.array(npy_records)
        
        frames_array = np.stack([npz_file[idx] for idx in list_IDs_temp], axis=0)
        logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))
        
        return frames_array
    
    def __get_records(self,list_IDs_temp):
        """Custom method - get the y labels
        """
        # Load the saved records
        df_records = pd.read_pickle(self.path_records)
        # Set the index to match
        df_records.index = df_records['timestamp']
        # Subset
        this_batch_steering = df_records.loc[list_IDs_temp]
        
        steering_values = this_batch_steering['steering_signal'].values
        
        #print(steering_values)
        steering_records_array = bin_Y(steering_values)
        
        #df_categorical_steering = df_records['steering_signal']
        #
        
        
        #records_array = df_records[]
        logging.debug("Generating {} records {}:".format(steering_records_array.shape[0],steering_records_array.shape))
        return steering_records_array
        
        #raise
    
    def __data_generation(self, list_IDs_temp):
        """Keras generator method - Generates data containing batch_size samples
        """
        # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)
        
        X = self.__get_npy_arrays(list_IDs_temp)
        y = self.__get_records(list_IDs_temp)
        
        # Generate data
        #for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            #y[i] = self.labels[ID]
        #    pass

        return X, y



#%%
    
def get_fig_as_npy(this_fig):
    raise
    """Take a matplotlib.figure.Figure and return it as a static npy RGB array 
    """
    canvas = mpl.backends.backend_agg.FigureCanvas(this_fig)
    canvas.draw() 
    width, height = this_fig.get_size_inches() * this_fig.get_dpi()
    width = int(width)
    height = int(height)
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return img




#%%
def plot_frames(records):
    """
    Render N records to analysis
    """
    font_label_box = {
        'color':'green',
        'size':16,
    }
    font_steering = {'family': 'monospace',
            #'color':  'darkred',
            'weight': 'normal',
            'size': 25,
            }
    
    fig=plt.figure(figsize=[20,18],facecolor='white')
    ROWS = 1
    COLS = 4
    NUM_IMAGES = ROWS * COLS

    for i,rec in enumerate(records):

                
        
        timestamp_string = rec['timestamp'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['timestamp'].microsecond))
        
        this_label = "{}\n{:0.2f} steering\n{:0.2f} throttle".format(rec['timestamp'],rec['steering_signal'],rec['throttle'])
        ax = fig.add_subplot(ROWS,COLS,i+1)
        ax.imshow(rec['frame'])
        #plt.title(str_label)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        t = ax.text(5,25,this_label,color='green',alpha=1)
        #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
        t.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))
        
        steer_cat = ''.join(['|' if v else '-' for v in linear_bin(rec['steering_signal'])])
        
        if 'steering_pred_signal' in df_records.columns:
            #steer_pred = ''.join(['◈' if v else ' ' for v in linear_bin(rec['steering_pred_signal'])])
            steer_pred = ''.join(['◈' if v else ' ' for v in linear_bin(rec['steering_pred_signal'])])
            
            #print(steer_pred)
            #ax.text(80,95,steer_pred,fontsize=30,horizontalalignment='center',verticalalignment='center',color='red')
            #ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
            #ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
        else: 
            steer_pred = ""
            
        hud_text =  steer_pred + "\n" + steer_cat
        
        #ax.text(80,105,steer_cat,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
        #ax.text(80,95,hud_text,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
        #ax.text(80,95,hud_text,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
        
        ax.text(80,105,steer_cat,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')        
        ax.text(80,105,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
        



#%%
def get_frames(path_frames,frame_ids):
    """From a list of ID strings, fetch the npy arrays from the zip. 
    """
    if type(frame_ids) == str:
        frame_ids = [frame_ids]
    npz_file=np.load(path_frames)
    frames_array = np.stack([npz_file[idx] for idx in frame_ids], axis=0)
    logging.debug("Returning frames array {}".format(frames_array.shape))
    return frames_array



#%%

#joined['steering_pred_cats'] = joined['steering_pred'].apply(linear_bin)
#joined['steering_pred_argmax'] = joined['steering_pred_cats'].apply(np.argmax)



def gen_one_record_frame(rec):
    """From a Record dictionary, create a single summary image of that timestep. 
    
    The figure has no border (full image)
    
    Show a data box with throttle and steering values. 
    Show also the predicted values, if available. 
    
    Show a steering widget to visualize the current steering signal. 
    Show also the predicted value, if available. 
    
    """
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
    fig = plt.figure(frameon=False,figsize=(HEIGHT_INCHES,WIDTH_INCHES))
    #fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rec['frame'])
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



#%%
def get_full_records(this_frames, this_df_records, this_indices):
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
    #record = dict()
    #record['frame'] 
#     OLD METHOD, VECTORIZED:
#         rec['frame'] = np.stack(this_frames[this_idx] for idx in this_idx], axis=0)
#         rec['steer'] = df_records[df_records['timestamp'].isin(this_idx)]['steering_signal'].values
#         rec['throttle'] = df_records[df_records['timestamp'].isin(this_idx)]['throttle_signal'].values
#         timestamp = df_records[df_records['timestamp'].isin(this_idx)]['timestamp'].values
#         rec['timestamp'] = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamp]
#         rec['steer_pred'] = y_pred_floats[y_pred_floats.index.isin(this_idx)]['steering_pred'].values 




#%%
##indices= sel_indices
def get_n_records(df_records, frames, indices):
    raise
    """
    """
    #this_frame = np.array[frames[idx] for idx in indices]
    these_frames = [frames[idx] for idx in indices]
    
    frame_array = np.stack([frames[idx] for idx in indices], axis=0)
    #this_steering = df_records[df_records['timestamp'] == idx]['steering_signal']
    these_steering = df_records[df_records['timestamp'].isin(indices)]['steering_signal'].values
    
    these_throttle = df_records[df_records['timestamp'].isin(indices)]['throttle_signal'].values
    
    timestamps = df_records[df_records['timestamp'].isin(indices)]['timestamp'].values
    #this_steering = df_records[idx]
    
    these_ts = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
    return frame_array,these_steering,these_throttle, these_ts
    

#frame, steering = get_record(df_records,frames, '1533666134582')



#%%

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


    this_return_dict['df_record'] = os.path.join(this_dir,'df_record.pck')
    
    if not os.path.exists(this_return_dict['df_record']):
        create_record_df(this_return_dict['json_record_zip'],this_return_dict['df_record'])
        assert os.path.exists(this_return_dict['df_record'])
    
    return this_return_dict

#%%
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
    
    
#%%
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



#%%
def write_video(frames,path_video_out, fps, width, height):
    """From a list of frames objects, generate a MP4. 
    
    The frames objects are dictionaries with NPY arrays and the timestamp.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    
    # This is extremely picky, and can fail (create empty file) with no warning !!
    writer = cv2.VideoWriter(path_this_video_out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))

    for this_frame_dict in frames:
        writer.write(this_frame_dict['array']) # Write out frame to video
    
    logging.debug("Wrote {} frames to {}".format(len(frames),path_video_out))
    
    writer.release()
    cv2.destroyAllWindows()
    
path_this_video_out = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,VIDEO_OUT_NAME+'.mp4')
frames_height = frames[0]['array'].shape[0]
frames_width = frames[0]['array'].shape[1]
write_video(frames,path_this_video_out, fps=30, width=frames_width, height=frames_height)


    
#%%
these_records = get_full_records(frames_npz, df_records, df_records.index)
#test = rec['frame']

#%% !!! REPLACE THE FRAMES WITH SALIENCY FRAMES IN EACH RECORD!!! 
for rec in these_records:
    rec['frame'] = saliency_img_dict[rec['timestamp_raw']]



#%% Reload the JPG back to NPY arrays

#path_frames_dir = path_video_frames_temp
def get_sorted_jpg_frames(path_frames_dir):
    """From a directory holding .jpg images, load as NPY, and sort on timestamp
    
    All arrays are stored in a dictionary with the timestamp for sorting.
    """
    # Look inside
    frames = list()
    frame_files_list = glob.glob(os.path.join(path_frames_dir,'*.jpg'))
    for this_img_path in frame_files_list:
        this_frame_dict = dict()
        #print(this_jpg)
        _, this_img_fname = os.path.split(this_img_path)
        timestamp = os.path.splitext(this_img_fname)[0]
        datetime_stamp = datetime.datetime.fromtimestamp(int(timestamp)/1000)
        datetime_str = datetime_stamp.isoformat()
        #img_bytes = this_img.read(this_img)
        #img_uint8 = plt.imread(this_img_path)
        img = cv2.imread(this_img_path)
        #img.shape
        ##img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)
        #img = cv2.imdecode(img_uint8, -1)
        #cv2.imdecode(img_uint8)
        #frames[datetime_str] = img
        this_frame_dict['array'] = img
        this_frame_dict['datetime_stamp'] = datetime_stamp
        this_frame_dict['datetime_str'] = datetime_str
        frames.append(this_frame_dict)

    # Sort the frames!
    frames = sorted(frames, key=lambda k: k['datetime_stamp'])
    logging.debug("Collected {} frames".format(len(frames)))

    return frames

with LoggerCritical():
    frames = get_sorted_jpg_frames(path_video_frames_temp)


#%% Render all to JPG
"""
"""
#%matplotlib qt
#

#%matplotlib qt
#plt.ioff()
df_records = pd.read_pickle(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',THIS_MODEL_ID+' predicted.pck'))

img_files_list = glob.glob(os.path.join(path_saliency_blend_frames,'*.png'))
saliency_img_dict = dict()
for f in img_files_list:
    _, fname_ext = os.path.split(f)
    fname, _ = os.path.splitext(fname_ext)
    saliency_img_dict[fname] = plt.imread(f)
    #print(f)

     


#%%
def write_video_figures(records,path_out):
    """For each record, generate a MPL Figure object. 
    
    Write each object to JPG. 
    """
    print('Start')
    for i,rec in enumerate(records):
        if i%10 == 0:
            print(i,"|",end="") #This doesn't show if capture is on! 
        with LoggerCritical():
            # Get a frame figure object
            record_figure = gen_one_record_frame(rec)
            
            # Save it to jpg
            this_fname = os.path.join(path_out,rec['timestamp_raw'] + '.jpg')
            logging.debug("Saved {}".format(this_fname))
            #print(fig)
            record_figure.savefig(this_fname)


if not os.path.exists(path_video_frames_temp): os.makedirs(path_video_frames_temp)

#

write_video_figures(these_records,path_video_frames_temp)

# -*- coding: utf-8 -*-



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

#%%OLD

def plot4(self,ts_string_indices, source_jpg_folder='jpg_images'):
    """
    Render N records to analysis
    """
    # Settings ############################################################
    font_label_box = {
        'color':'green',
        'size':16,
    }
    font_steering = {'family': 'monospace',
            #'color':  'darkred',
            'weight': 'normal',
            'size': 25,
            }
    ROWS = 1
    COLS = 4
    NUM_IMAGES = ROWS * COLS
    
    # Figure ##############################################################
    fig=plt.figure(figsize=[20,18],facecolor='white')

    
    for i,ts_string_index in enumerate(ts_string_indices):
        rec = self.df.loc[ts_string_index]

        timestamp_string = rec['datetime'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['datetime'].microsecond))
        
        if 'steering_pred_signal' in self.df.columns:
            this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,
                          rec['steering_signal'],rec['steering_pred_signal'],rec['throttle_signal'])
        else: 
            this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string,rec['steering_signal'],rec['throttle_signal'])
            
        ax = fig.add_subplot(ROWS,COLS,i+1)

        # Main Image ##########################################################
        jpg_path = os.path.join(self.path_dataset,source_jpg_folder,ts_string_index+'.jpg')
        assert os.path.exists(jpg_path)
        img = mpl.image.imread(jpg_path)
        ax.imshow(img)
        #plt.title(str_label)
        
        # Data box ########################################################
        
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_yaxis().set_visible(False)
        t = ax.text(5,25,this_label,color='green',alpha=1)
        #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
        t.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))
        
        # Steering widget HUD #################################################
        # Steering HUD: Actual steering signal
        steer_actual = ''.join(['|' if v else '-' for v in self.linear_bin(rec['steering_signal'])])
        text_steer = ax.text(80,105,steer_actual,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
        # Steering HUD: Predicted steering angle
        if 'steering_pred_signal' in self.df.columns:
            steer_pred = ''.join(['◈' if v else ' ' for v in self.linear_bin(rec['steering_pred_signal'])])
            text_steer_pred = ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')




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