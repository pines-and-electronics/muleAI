
#%% Partition

msk = np.random.rand(len(df_records)) < 0.8
partition = dict()
partition['train'] = df_records.index[msk].values
partition['validation'] = df_records.index[~msk].values
#labels = range(15)

#%% Generator

generator_params = {'dim': (160,120),
          'batch_size': 32,
          'n_classes': 15,
          'n_channels': 3,
          'shuffle': True,
          'path_frames':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'),
          'path_records':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'),
         }

training_generator = MuleDataGenerator(partition['train'], **generator_params)
validation_generator = MuleDataGenerator(partition['validation'], **generator_params)
logging.debug("**")
logging.debug("Data Generators: {} samples over batch size {} yields ~{} batches: {} / {} train/val ".format(len(df_records),
                                                                                   generator_params['batch_size'],
                                                                                   math.ceil(len(df_records)/generator_params['batch_size']),
                                                                                   len(training_generator),
                                                                                  len(validation_generator)))

# In[16]:

first_batch = validation_generator[0]
last_batch = validation_generator[len(validation_generator)]

# In[17]:


def baseline_steering_model():
    model = ks.models.Sequential()
    model.add(ks.layers.Conv2D(24, (5,5), strides=(2, 2), activation = "relu", input_shape=(120,160,3)))
    model.add(ks.layers.Conv2D(32, (5,5), strides=(2, 2), activation = "relu"))
    model.add(ks.layers.Conv2D(64, (5,5), strides=(2, 2), activation = "relu"))
    model.add(ks.layers.Conv2D(64, (3,3), strides=(2, 2), activation = "relu"))
    model.add(ks.layers.Conv2D(64, (3,3), strides=(1, 1), activation = "relu"))
    model.add(ks.layers.Flatten()) # This is just a reshape!
    model.add(ks.layers.Dense(100,activation="relu"))
    model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(50,activation="relu"))
    model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(15, activation='softmax', name='angle_out'))
    return model
blmodel = baseline_steering_model()


# In[18]:


blmodel.summary()


# In[19]:


blmodel.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=[ks.metrics.categorical_accuracy]
               )


# # TRAIN

# In[20]:


EPOCHS = 10
with LoggerCritical():
    history = blmodel.fit_generator(generator=training_generator,
                      validation_data=validation_generator,
                      use_multiprocessing=True,
                      workers=6,
                      epochs=EPOCHS,
                      verbose=1,)
history_dict = history.__dict__


#%% Make predictions and augment the records frame 

df_records = get_predictions(blmodel, frames_npz, df_records)
sum(df_records['steering_pred_argmax'] == df_records['steering_signal_argmax'])/len(df_records)
#df_records.columns
#%%
records = get_full_records(frames, df_records, y_pred_floats, [sel_indices[0]])


# In[ ]:


# Predict the raw probability vectors
def raw_predictions(model,frames):
    frame_array = np.stack([frames[idx] for idx in frames], axis=0)
    y_pred_categorical = model.predict(frame_array)
    #print(y_pred_categorical)
    #y_pred = unbin_Y(y_pred_categorical)
    df_pred = pd.DataFrame(y_pred_categorical,index = [idx for idx in frames])
    df_pred.sort_index(inplace = True)
    return df_pred

df_ypred_probs = raw_predictions(blmodel,frames)
df_ypred_probs.head()


# In[ ]:


df_records['steering_signal'].head()


# # TODO: JOIN The data frames to ensure alignment! 

# # TODO: BROKEN!!!

# In[ ]:


get_ipython().run_cell_magic('script', 'false', '# Convert probability to the predicted argmax vector (one-hot)\ndef argmax_predictions(raw_probs):\n    # Get the argmax of each row, as a 1D array\n    argmaxxes = np.argmax(raw_probs.values,axis=1)\n    # Initialize the output array\n    arr = np.zeros((len(argmaxxes),15))\n    \n    # Iterate and set the max probability to 1, else 0\n    for i,argmax in enumerate(argmaxxes):\n        arr[i][argmax] = 1\n    #frame_array = np.stack([frames[idx] for idx in frames], axis=0)\n    #y_pred_categorical = model.predict(frame_array)\n    #print(y_pred_categorical)\n    #y_pred = unbin_Y(y_pred_categorical)\n    return pd.DataFrame(arr,index = raw_probs.index)\n\ndf_ypred_onehot = argmax_predictions(df_ypred_probs)\ndf_ypred_onehot.head()')


# In[ ]:


get_ipython().run_cell_magic('script', 'false', '# Convert the category back to a floating point steering value\ndef categorical_predictions(y_pred_raw):\n    y_pred_cats = unbin_Y(y_pred_raw.values)\n    #pd.DataFrame(y_pred_cats,index = [idx for idx in frames])\n    return pd.DataFrame({"steering_pred":y_pred_cats},index = [idx for idx in frames])   \ny_pred_floats = categorical_predictions(df_ypred_onehot)\ny_pred_floats.head()')


# # Manual and Keras metric calculations

# In[ ]:


res = df_records.loc[:,['steering_signal',]]
joined = df_records.loc[:,['steering_signal','throttle_signal']].join(y_pred_floats)


# In[ ]:


#joined.head()


# In[ ]:



# In[ ]:


joined.head()


# In[ ]:


# Accuracy!
sum(joined['steering_signal_cats_argmax'] == joined['steering_pred_argmax'])/len(joined['steering_pred_argmax'])


# In[ ]:


# Mean Squared Error from y_pred_floats
targets = df_records['steering_signal'].values
predictions = y_pred_floats['steering_pred'].values
N = predictions.shape[0]
sums_cumul = list()
for y_actual, y_pred in zip(targets,predictions):
    part_sum = (y_actual-y_pred)**2
    #print(y_actual, " - ",y_pred, "=",part_sum)
    sums_cumul.append(part_sum)
print("Total sum:",sum(sums_cumul))
print("MSE:",sum(sums_cumul)/N)


# In[ ]:


# Categorical accuracy (= number correctly categorized)
# Sum the correctly matching categories
targets = df_records['steering_signal'].values
predictions = y_pred_floats['steering_pred'].values
N = predictions.shape[0]
sums_cumul = list()
for y_actual, y_pred in zip(targets,predictions):
    y_actual_cats = linear_bin(y_actual)
    y_pred_cats = linear_bin(y_pred)
    #print(y_actual_cats)
    #print(y_pred_cats)
    sums_cumul.append(all(y_actual_cats == y_pred_cats))
    
    #part_sum = (y_actual-y_pred)**2
    #print(y_actual, " - ",y_pred, "=",part_sum)
    #sums_cumul.append(part_sum)
print("Total sum:",sum(sums_cumul))
print("Categorical Accuracy: {:0.1f}%".format(sum(sums_cumul)/N*100))


# In[ ]:


get_ipython().run_cell_magic('script', 'false', 'sess = tf.InteractiveSession()\nwith sess.as_default():\n    ks_categorical_accuracy = ks.metrics.categorical_accuracy(bin_Y(df_records[\'steering_signal\']), bin_Y(y_pred_floats[\'steering_pred\'])).eval()\n    ks_mean_squared_error = ks.metrics.mean_squared_error(bin_Y(df_records[\'steering_signal\']), bin_Y(y_pred_floats[\'steering_pred\'])).eval()\n\n    ks_categorical_crossentropy = ks.losses.categorical_crossentropy(ks.backend.constant(bin_Y(df_records[\'steering_signal\'])), \n                                                                          ks.backend.constant(bin_Y(y_pred_floats[\'steering_pred\']))).eval()\nprint("categorical_accuracy", sum(ks_categorical_accuracy)/N)\nprint("mean_squared_error",sum(ks_mean_squared_error)/N)\nprint("categorical_crossentropy",sum(ks_categorical_crossentropy)/N)')


# # Epoch progress plot

# In[ ]:


#model_title = "10 Epochs"
def plot_hist(history_dict, accuracy_name, model_title):
    #fig = plt.figure(figsize=(5,4))
    #fig=plt.figure(figsize=(20, 10),facecolor='white')

    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5),sharey=False,facecolor='white')
    bgcolor = '0.15'
    bgcolor = 'white'
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5),sharey=False,facecolor=bgcolor)
    
    ax1.plot(history_dict['epoch'],  history_dict['history']['loss'],label="Train")
    ax1.plot(history_dict['epoch'],  history_dict['history']['val_loss'],label="CV")
    ax1.set_title("Loss function development - Training set vs CV set")
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Values')
    
    ax2.plot(history_dict['epoch'],  history_dict['history'][accuracy_name],label="Train")
    ax2.plot(history_dict['epoch'],  history_dict['history']['val_'+accuracy_name],label="CV")
    ax2.set_title("Accuracy development - Training set vs CV set")
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Values')
    
    plt.suptitle(model_title, fontsize=16)
    
    plt.show()

plot_hist(history_dict,'categorical_accuracy',model_title="")


# In[ ]:


def get_n_predictions(model,frames,indices):
    """Given a model and the input frames, select a subset (indices)
    and return the predictions.
    """
    these_frames = [frames[idx] for idx in indices]
    frame_array = np.stack([frames[idx] for idx in indices], axis=0)
    y_pred_categorical = model.predict(frame_array)
    #print(y_pred_categorical)
    y_pred = unbin_Y(y_pred_categorical)
    return y_pred


# # Plot some results

# In[ ]:


def get_n_records(df_records, frames, indices):
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


# # Utility to get a number of dictionary records

# In[ ]:


#'1533666054613'


# In[ ]:


#[sel_indices[0]]


# In[ ]:


# Image parameters
ROWS = 4
COLS = 4
NUM_IMAGES = ROWS * COLS


# In[ ]:


sel_indices = df_records.sample(NUM_IMAGES)['timestamp'].values
records = get_full_records(frames, df_records, y_pred_floats, sel_indices)


# In[ ]:





# In[ ]:



