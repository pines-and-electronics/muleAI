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
logging.debug("Loaded {} frames from {}".format(len(frames_npz),THIS_DATASET))
logging.debug("Loaded {} records from {}".format(len(df_records),THIS_DATASET))

#%%
df_records['steering_signal'] = df_records['steering_signal'].apply(lambda x: x*-1)
logging.debug("Steering signal inverterted".format())

#%% Expand the steering signal
df_records['steering_signal_cats'] = df_records['steering_signal'].apply(linear_bin)
df_records['steering_signal_argmax'] = df_records['steering_signal_cats'].apply(np.argmax)


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


#%% Plot a few samples
# TURN OFF PLOTTING
%matplotlib inline

# Right turn
these_indices = df_records[df_records['steering_signal'] > 0.9].sample(4)['timestamp'].tolist()
these_records = get_full_records(frames_npz, df_records, these_indices)
#these_frames = 
for rec in these_records:
    gen_one_record_frame(rec)
plot_frames(these_indices,df_records, frames_npz)

# Left turn
these_indices = df_records[df_records['steering_signal'] < -0.9].sample(4)['timestamp'].tolist()
plot_frames(these_indices,df_records, frames_npz)

# Straight
these_indices = df_records[(df_records['steering_signal'] > -0.1) & (df_records['steering_signal'] < 0.1)].sample(4)['timestamp'].tolist()
plot_frames(these_indices,df_records, frames_npz)

#%%
# TURN OFF PLOTTING
%matplotlib qt
plt.ioff()


#records = get_full_records(frames_npz, df_records, df_records.index[0:3])
#for r in records:
#    this = gen_one_record_frame(r,"")
#type(this)




