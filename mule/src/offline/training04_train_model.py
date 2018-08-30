#%% Partition the data

msk = np.random.rand(len(df_records)) < 0.8
partition = dict()
partition['train'] = df_records.index[msk].values
partition['validation'] = df_records.index[~msk].values
#labels = range(15)

#%% Generators

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

#%% Test the generator class

first_batch = validation_generator[0]
last_batch = validation_generator[len(validation_generator)]

#%% The model

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
blmodel.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=[ks.metrics.categorical_accuracy]
               )
blmodel.summary()

#%% TRAIN

EPOCHS = 10
with LoggerCritical():
    history = blmodel.fit_generator(generator=training_generator,
                      validation_data=validation_generator,
                      use_multiprocessing=True,
                      workers=6,
                      epochs=EPOCHS,
                      verbose=1,)

history_dict = history.__dict__

this_timestamp = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
logging.debug("Finished training model {}".format(this_timestamp))

#%% Make predictions and augment the records frame with predicted values

df_records = get_predictions(blmodel, frames_npz, df_records)
# The manually calculated accuracy: 
#sum(df_records['steering_pred_argmax'] == df_records['steering_signal_argmax'])/len(df_records)
sum(df_records['steering_pred_argmax'] == df_records['steering_signal_argmax'])/len(df_records)
# Invert the steering! WHY>?
df_records['steering_pred_signal'] = df_records['steering_pred_signal'].apply(lambda x: x*-1)
logging.debug("Steering signal inverterted - WHY?".format())

#%% =============================================================================
# Plot analysis of model
# =============================================================================
# Plot training history
#%matplotlib inline
#model_title = "10 Epochs"

plot_hist(history_dict,'categorical_accuracy',model_title="")

# Plot a few samples
# TURN OFF PLOTTING
#%matplotlib inline

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


#%% =============================================================================
# Saving the model
# =============================================================================
# Save the model weights, architecture, configuration, and state
# serialize weights to HDF5
path_model_h5 = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' model.h5')
model_dir = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
blmodel.save(path_model_h5)
logging.debug("Saved complete model hd5 to disk".format())

# Save just the architecture to JSON
path_model_json = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' model.json')
model_json = blmodel.to_json()
with open(path_model_json, "w") as json_file:
    json_file.write(model_json)
logging.debug("Saved model configuration json to disk".format())

# Save the history to JSON
path_history_json = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' history.json')
with open(path_history_json, "w") as json_file:
    if 'model' in history_dict: 
        history_dict.pop('model')
    json.dump(history_dict,json_file)
logging.debug("Saved model history json to disk".format())

# Save the predicted dataframe
path_records_json = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' predicted.pck')
df_records.to_pickle(path_records_json)

#%%
if 0:
    test = ks.models.load_model(path_model_h5)
    
    test.summary()
    test.metrics
    dir(test)






#%% Manual and Keras metric calculations
if 0:
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




## In[ ]:
#
#
## Image parameters
#ROWS = 4
#COLS = 4
#NUM_IMAGES = ROWS * COLS
#
#
## In[ ]:
#
#
#sel_indices = df_records.sample(NUM_IMAGES)['timestamp'].values
#records = get_full_records(frames, df_records, y_pred_floats, sel_indices)



