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

#%% Invert the steering!, +1 = right turn
df_records['steering_signal'] = df_records['steering_signal'].apply(lambda x: x*-1)
logging.debug("Steering signal inverterted".format())

#%% Expand the steering signal
df_records['steering_signal_cats'] = df_records['steering_signal'].apply(linear_bin)
df_records['steering_signal_argmax'] = df_records['steering_signal_cats'].apply(np.argmax)

#%% Histogram: Steering
fig=plt.figure(figsize=[10,5],facecolor='white')
hist_steering = df_records['steering_signal'].hist()

#%% Histogram: Throttle
fig=plt.figure(figsize=[10,5],facecolor='white')
hist_throttle = df_records['throttle_signal'].hist()
plot_url = py.plot_mpl(fig)

#%% Ignore throttle = 0 images
if 0:
    original_size = len(df_records)
    df_records = df_records[df_records['throttle_signal']>0.001]
    logging.debug("{} no-throttle records ignored, {} remain".format(original_size-len(df_records), len(df_records)))

    fig=plt.figure(figsize=[10,5],facecolor='white')
    hist_steering = df_records['steering_signal'].hist()


#%% Plot a few samples
# TURN OFF PLOTTING
%matplotlib inline

# Right turn
these_indices = df_records[df_records['steering_signal'] > 0.9].sample(4)['timestamp'].tolist()
these_records = get_full_records(frames_npz, df_records, these_indices)
plot_frames(these_records)

# Left turn
these_indices = df_records[df_records['steering_signal'] < -0.9].sample(4)['timestamp'].tolist()
these_records = get_full_records(frames_npz, df_records, these_indices)
plot_frames(these_records)

# Straight
these_indices = df_records[(df_records['steering_signal'] > -0.1) & (df_records['steering_signal'] < 0.1)].sample(4)['timestamp'].tolist()
these_records = get_full_records(frames_npz, df_records, these_indices)
plot_frames(these_records)




