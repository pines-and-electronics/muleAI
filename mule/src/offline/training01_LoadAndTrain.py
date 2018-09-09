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
#DATASET_ID = "20180907 193757"

assert os.path.exists(LOCAL_PROJECT_PATH)

#%% ===========================================================================
# Load dataset
# =============================================================================
ds = AIDataSet(LOCAL_PROJECT_PATH,DATASET_ID)
ds.process_time_steps()
if True: 
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
dsm.instantiate_generators(MuleDataGenerator)
dsm.instantiate_model()
dsm.model.summary()
dsm.instantiate_callbacks()
dsm.callback_list
dsm.train_model(50)
dsm.make_predictions()

