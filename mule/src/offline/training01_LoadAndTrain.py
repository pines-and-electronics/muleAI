"""
Manage and analyze the Data Set directory.
Iterate over each Data Set and check which data elements have been created. 

NOT Standalone script. 

"""
#%% Instantiate and load the dataset

LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
#DATASET_ID = "20180907 193306"
DATASET_ID = "20180907 184100 BENCHMARK1 TRG"
DATASET_ID = "20180907 191303 BENCHMARK1 AUTO"
DATASET_ID = "20180907 180022"
DATASET_ID = "20180909 215233"
DATASET_ID = "20180910 181306"
DATASET_ID = "20180910 192658"

#DATASET_ID = "20180909 213918"

#/home/batman/MULE DATA/
#DATASET_ID = "20180907 193757"

assert os.path.exists(LOCAL_PROJECT_PATH)

#%% ===========================================================================
# Load dataset
# =============================================================================
ds = AIDataSet(LOCAL_PROJECT_PATH,DATASET_ID)
ds.process_time_steps()
if False: 
    ds.write_jpgs(dir_jpgs="jpg_images", overwrite=False)

#%% ===========================================================================
# Mask values
# =============================================================================

ds.mask_first_Ns(2)
ds.mask_last_Ns(2)
ds.mask_null_throttle(0.1)   

#%% ===========================================================================
# Summary plots
# =============================================================================
with NoPlots():
    plotter = DataSetPlotter()
    plotter.histogram_steering(ds)
    plotter.histogram_throttle(ds)
    with LoggerCritical():
        sample_fig = plotter.plot_sample_frames(ds)
    
    plotter.boxplots_time(ds)

#%% ===========================================================================
# Frames: to /Video Frames and /Video Frames.mp4
# =============================================================================
if False:
    ds.write_frames(overwrite=False)
    
    PATH_INPUT_JPGS = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames')
    PATH_OUTPUT_FILE = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames.mp4')
    
    vidwriter = VideoWriter(PATH_INPUT_JPGS,PATH_OUTPUT_FILE,fps=24)
    vidwriter.write_video()


#%% Start CUDA and Training
ks.backend.clear_session()
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
for dev in devices:
    print(dev.name)

#%% ===========================================================================
# Train model
# =============================================================================

# New Model_ID
THIS_MODEL_ID = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
#THIS_MODEL_ID = '20180909 160218'
dsm=ModelledData(ds,THIS_MODEL_ID)
dsm.model_folder_empty
dsm.generate_partitions()
#trained_dataset.list_models()
#dsm.instantiate_generators(MuleDataGeneratorBlackWhite)
#dsm.instantiate_model(model_name="blackwhite_steering_model")

dsm.instantiate_generators(MuleDataGenerator)
dsm.instantiate_model(model_name="baseline_steering_model")

dsm.model.summary()
#raise
dsm.instantiate_callbacks()
dsm.callback_list
dsm.train_model(50)
dsm.make_predictions()


#%% TEST GEN
if 0:
    generator_params = {'dim': (160,120),
              'batch_size': 64,
              'n_classes': 15,
              'n_channels': 3,
              'shuffle': True,
              #'path_frames':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'),
              #'path_records':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'),
             }
    
    training_generator = MuleDataGeneratorBlackWhite(dsm.partition['train'], dsm.ds, **generator_params)
    first_batch = training_generator[0]
    last_batch = training_generator[len(training_generator)]