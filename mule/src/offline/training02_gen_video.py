
#%% Render all to JPG
"""
"""
path_video_frames_temp = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'temp_frames')
if not os.path.exists(path_video_frames_temp): os.makedirs(path_video_frames_temp)

these_records = get_full_records(frames_npz, df_records, df_records.index)
for i,rec in enumerate(these_records):
    if i%10 == 0:
        print(i,"|",end="") #This doesn't show if capture is on! 
    with LoggerCritical():
        # Get a frame figure object
        record_figure = gen_one_record_frame(rec)
        
        # Save it to jpg
        this_fname = os.path.join(path_video_frames_temp,rec['timestamp_raw'] + '.jpg')
        logging.debug("Saved {}".format(this_fname))
        #print(fig)
        record_figure.savefig(this_fname)
        
        
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

#%%
def write_video(frames,path_video_out, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    
    # This is extremely picky, and can fail (create empty file) with no warning !!
    writer = cv2.VideoWriter(path_this_video_out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))

    for this_frame_dict in frames:
        writer.write(this_frame_dict['array']) # Write out frame to video
    
    logging.debug("Wrote {} frames to {}".format(len(frames),path_video_out))
    
    writer.release()
    cv2.destroyAllWindows()
    

path_this_video_out = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'video_with_signals.mp4')
frames_height = frames[0]['array'].shape[0]
frames_width = frames[0]['array'].shape[1]
write_video(frames,path_this_video_out, fps=30, width=frames_width, height=frames_height)

#%% TEST VIDEO WITH RANDOM NOISE
if 0:
    frames[0]['array'].shape
    frames[0]['array'].dtype
    this_frame.shape
    this_frame.dtype

    path_this_video_out = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'test.mp4')
    
    H = 480
    
    W = 640
    
    
    writer = cv2.VideoWriter(path_this_video_out, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W,H))
    for frame in range(400):
        this_frame = np.random.randint(0, 255, (H,W,3)).astype('uint8')
        this_frame = frames[0]['array']
        writer.write(this_frame)
    writer.release()
