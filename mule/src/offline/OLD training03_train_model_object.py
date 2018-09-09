"""



Needs the AIDataSet

"""
import json

ks.backend.clear_session()
#%% Get a new model folder, if necessary
THIS_MODEL_ID = datetime.datetime.now().strftime("%Y%m%d %H%M%S")

#%% Check CUDA
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
for dev in devices:
    print(dev.name)

#%%
raise Exception("Manual execution below")

#%% RELOAD A MODEL
if False:
    LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
    THIS_DATASET = "20180904 192907"
    THIS_DATASET = "20180907 174134"
    THIS_DATASET = "20180907 180022"
    
    #THIS_MODEL_ID = 'model 20180906 154310'
    trds = ModelledDataSet(LOCAL_PROJECT_PATH,THIS_DATASET,THIS_MODEL_ID)
    trds.load_best_model()
    trds.model.summary()
    trds.make_predictions()


        
#%% CREATE NEW MODEL AND TRAIN
THIS_DATASET = "20180907 184100"
THIS_DATASET = "20180907 193306"

LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]

#trained_dataset = ModelledDataSet(LOCAL_PROJECT_PATH,THIS_DATASET,'model ' + THIS_MODEL_TIMESTAMP)
THIS_MODEL_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d %H%M%S")

trds = ModelledDataSet(LOCAL_PROJECT_PATH,THIS_DATASET,'model ' + THIS_MODEL_TIMESTAMP)
trds.model_folder_empty
trds.generate_partitions()
#trained_dataset.list_models()
trds.instantiate_generators(MuleDataGenerator)
trds.instantiate_model()
trds.model.summary()
trds.instantiate_callbacks()
trds.train_model(30)
trds.make_predictions()
raise

#%% Plot analysis of model

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


#%% Saving the model
## Save the model weights, architecture, configuration, and state
## serialize weights to HDF5
#path_model_h5 = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' model.h5')
#blmodel.save(path_model_h5)
#logging.debug("Saved complete model hd5 to disk".format())
#
## Save just the architecture to JSON
#path_model_json = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' model.json')
#model_json = blmodel.to_json()
#with open(path_model_json, "w") as json_file:
#    json_file.write(model_json)
#logging.debug("Saved model configuration json to disk".format())
#
## Save the history to JSON
#path_history_json = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' history.json')
#with open(path_history_json, "w") as json_file:
#    if 'model' in history_dict: 
#        history_dict.pop('model')
#    json.dump(history_dict,json_file)
#logging.debug("Saved model history json to disk".format())
#
## Save the predicted dataframe
#path_records_json = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'model',this_timestamp + ' predicted.pck')
#df_records.to_pickle(path_records_json)




