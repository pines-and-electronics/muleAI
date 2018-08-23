"""
Select a single data set for training
Load the records and frames>
Plot some analysis
""" 

#%% Select the directory

LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)

#%% Select the data set

#THIS_DATASET = '20180807 201756'
THIS_DATASET = '20180801 160056'
df_records = pd.read_pickle(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'))
frames_npz=np.load(os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'))
df_records.index = df_records['timestamp']
df_records.sort_index(inplace=True)
df_records['steering_signal'] = df_records['steering_signal'].apply(lambda x: x*-1)


#%% Load the frames
#
#
#df_small = df_records.head(10)
##
#len(df_small),
#
#df_small['frame'] = np.empty()
#
#frames = np.nan(len(df_small))
#
#for idx in df_small.index:
#    print(idx)
#    df_small.loc[idx,'frame'] = get_frames(path_frames,idx)
#    

#path_frames = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip')
#df_records.index[0]
#frames = get_frames(path_frames,df_records.index[0])
#frames = get_frames(path_frames,df_records.index)

#%% Histogram: Steering

fig=plt.figure(figsize=[10,5],facecolor='white')
hist_steering = df_records['steering_signal'].hist()

#%%
#plot_url = py.plot_mpl(fig)

#%% Histogram: Throttle

fig=plt.figure(figsize=[10,5],facecolor='white')
hist_throttle = df_records['throttle_signal'].hist()
plot_url = py.plot_mpl(fig)


# # Subset only of positive throttle

#%% Ignore throttle = 0 images
if 0:
    original_size = len(df_records)
    df_records = df_records[df_records['throttle_signal']>0.001]
    logging.debug("{} no-throttle records ignored, {} remain".format(original_size-len(df_records), len(df_records)))

    fig=plt.figure(figsize=[10,5],facecolor='white')
    hist_steering = df_records['steering_signal'].hist()


#%% Test the record selection

#sel_indices = df_records.sample(5)['timestamp'].values
#sel_frames, sel_steerings, sel_throttles, sel_ts = get_n_records(df_records, frames_npz, sel_indices)


#%%
#import matplotlib.pyplot as plt
#import plotly.tools as tls

#mpl_fig = plt.figure()
# --> your matplotlib methods <--

#plotly_fig = tls.mpl_to_plotly(mpl_fig)


# In[12]:


#sel_indices = list()
#sel_indices += df_records[df_records['steering_signal'] > 0.9].sample(4)['timestamp'].tolist()

def plot_frames(sel_indices,df_records, frames_npz):
    """
    Plot 
    """
    fig=plt.figure(figsize=[20,18],facecolor='white')
    ROWS = 1
    COLS = 4
    NUM_IMAGES = ROWS * COLS
    #sel_indices = df_records.sample(NUM_IMAGES)['timestamp'].values
    sel_frames, sel_steerings, sel_throttles, these_ts = get_n_records(df_records, frames_npz, sel_indices)
    
    for i,record in enumerate(zip(sel_frames, sel_steerings,sel_throttles,these_ts)):
        this_frame, this_steer, this_throttle,sel_ts = record
        
        steer_cat = ''.join(['|' if v else '-' for v in linear_bin(this_steer)])
        timestamp_string = sel_ts.strftime("%D %H:%M:%S.") + "{:.2}".format(str(sel_ts.microsecond))
        
        this_label = "{}\n{:0.2f} steering\n{:0.2f} throttle".format(timestamp_string,this_steer,this_throttle)
        y = fig.add_subplot(ROWS,COLS,i+1)
        y.imshow(this_frame)
        #plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        t = y.text(5,25,this_label,color='green',alpha=1)
        #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
        t.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))
        y.text(80,105,steer_cat,fontsize=30,horizontalalignment='center',verticalalignment='center',color='green')
        #plt.title()

# Right turn
these_indices = df_records[df_records['steering_signal'] > 0.9].sample(4)['timestamp'].tolist()
plot_frames(these_indices,df_records, frames_npz)

# Left turn
these_indices = df_records[df_records['steering_signal'] < -0.9].sample(4)['timestamp'].tolist()
plot_frames(these_indices,df_records, frames_npz)

# Straight
these_indices = df_records[(df_records['steering_signal'] > -0.1) & (df_records['steering_signal'] < 0.1)].sample(4)['timestamp'].tolist()
plot_frames(these_indices,df_records, frames_npz)


#plot_url = py.plot_mpl(fig)
#tls.mpl_to_plotly(fig)
# # Data generator and utilities

#%%





#%%
def get_full_records(this_frames, this_df_records, this_indices):
    #assert type(this_indices) == list or type(this_indices) == np.ndarray
    """Given a list of indices (timestamps), return a list of records
    
    frames:
        The frame images as a numpy array
    df_records:
        The steering at these times as a float
        The throttle at these times as a float
        The timestep as a datetime 
        The predicted steering at these times
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
        print()
        rec['timestamp'] = datetime.datetime.fromtimestamp(int(rec['timestamp_raw'])/1000)
        if 'steering_pred_signal' in df_records.columns:
            #'steering_signal' in df_records.columns
            rec['steer_pred'] = df_records.loc[this_idx]['steering_pred_signal']

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
records = get_full_records(frames_npz, df_records, df_records.index[0:10])


#%%
def gen_one_record_frame(rec, save_folder_path):
    
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
    steer_actual = ''.join(['x' if v else '-' for v in linear_bin(rec['steering_signal'])])
    text_steer = ax.text(80,105,steer_actual,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
    
    if 'steering_pred_signal' in df_records.columns:
        steer_pred = ''.join(['◈' if v else ' ' for v in linear_bin(rec['steer_pred'])])
        text_steer_pred = ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
    
    

    this_fname = os.path.join(save_folder_path,rec['timestamp_raw'] + '.jpg')
    logging.debug("Saved {}".format(this_fname))
    
    fig.savefig(this_fname)


# In[ ]:
path_video_frames_temp = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'temp_frames')
if not os.path.exists(path_video_frames_temp): os.makedirs(path_video_frames_temp)
for i,index in enumerate(df_records.index):
    if i%10 == 0:
        print(i) #This doesn't show if capture is on! 
    with LoggerCritical():
        this_rec = get_full_records(frames_npz, df_records, [index])[0]
        gen_one_record_frame(this_rec,save_folder_path = path_video_frames_temp)
        
#%%
#path_frames_dir = path_video_frames_temp
def get_frames(path_frames_dir):
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
    frames = get_frames(path_video_frames_temp)

#%%

#%%

def write_video(frames,path_video_out, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    #fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
    writer = cv2.VideoWriter(path_this_video_out, cv2.VideoWriter_fourcc(*"MJPG"), 30,(width,height))

    #fourcc = cv2.VideoWriter_fourcc(*'DIVX') # Be sure to use lower case
    #writer = cv2.VideoWriter(path_video_out, fourcc, fps, (width,height))
    #writer = cv2.VideoWriter(path_video_out, fourcc, 20, (width,height))
    #-1
    #writer=cv2.VideoWriter("test1.avi", cv2.CV_FOURCC(*''), 25, (640,480))
    
    for this_frame_dict in frames:
            
        #image_path = os.path.join(dir_path, image)
        frame = this_frame_dict['array']
        
        writer.write(frame) # Write out frame to video
    
        #cv2.imshow('video',frame)
        #if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        #    break
    logging.debug("Wrote {} frames to {}".format(len(frames),path_video_out))
    
    writer.release()
    cv2.destroyAllWindows()
path_this_video_out = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'video_with_signals.mp4')
#path_this_video_out = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'video_with_signals.avi')
#write_video(frames[0:10],path_this_video_out)
write_video(frames,path_this_video_out, fps=10, width=438, height=585)

frames[0]['array'].shape
frames[0]['array'].dtype
this_frame.shape
this_frame.dtype

#%%
path_this_video_out = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'test.mp4')

H = 480
H = 438
W = 640
W = 585

writer = cv2.VideoWriter(path_this_video_out, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W,H))
for frame in range(400):
    this_frame = np.random.randint(0, 255, (H,W,3)).astype('uint8')
    this_frame = frames[0]
    writer.write(this_frame)
writer.release()




