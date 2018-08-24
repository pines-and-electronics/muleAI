#%%
if 0: 
    raise
    def get_full_records(this_frames, this_df_records, this_y_pred_floats, this_indices):
        assert type(this_indices) == list or type(this_indices) == np.ndarray
        """Given a list of indices (timestamps), return a list of records
        
        frames:
            The frame images as a numpy array
        df_records:
            The steering at these times as a float
            The throttle at these times as a float
            The timestep as a datetime 
        y_pred_floats:
            The predicted steering at these times
        """
        records = list()
        for this_idx in this_indices:
            #print(this_idx)
            rec = dict()
            rec['frame'] = this_frames[this_idx]
            rec['steer'] = df_records.loc[this_idx]['steering_signal']
            #print(rec['steer'])
            rec['throttle'] = df_records.loc[this_idx]['throttle_signal']
            rec['timestamp_raw'] = df_records.loc[this_idx]['timestamp']
            print()
            rec['timestamp'] = datetime.datetime.fromtimestamp(int(rec['timestamp_raw'])/1000)
            rec['steer_pred'] = y_pred_floats.loc[this_idx]['steering_pred']
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
    #




#%%
def gen_sample_frames(records):
    raise OBSELETE
    fig=plt.figure(figsize=[20,18],facecolor='white')

    for i,rec in enumerate(records):
        font_label_box = {
            'color':'green',
            'size':16,
        }
        font_steering = {'family': 'monospace',
                #'color':  'darkred',
                'weight': 'normal',
                'size': 25,
                }
        steer_actual = ''.join(['x' if v else '-' for v in linear_bin(rec['steer'])])
        steer_pred = ''.join(['◈' if v else ' ' for v in linear_bin(rec['steer_pred'])])
        timestamp_string = rec['timestamp'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['timestamp'].microsecond))

        this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,rec['steer'],rec['steer_pred'],rec['throttle'])
        y = fig.add_subplot(ROWS,COLS,i+1)
        y.imshow(rec['frame'])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        t = y.text(5,25,this_label,color='green',alpha=1)
        t.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))

        y.text(80,105,steer_actual,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='green')
        y.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')

gen_sample_frames(records)



#%%



#%%
#these_frames = 
frames = list()
for rec in these_records:
    frame = gen_one_record_frame(rec)
    #frames.append(frame)
    this_np_frame = get_fig_as_npy(frame)
    fig = plt.figure(frameon=False,figsize=(HEIGHT_INCHES,WIDTH_INCHES))
    fig = plt.figure()
    #fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(this_np_frame)
    
    
type(frame)
dir(frame)
frame.axes[0]
di
#plot_frames(these_indices,df_records, frames_npz)


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

fig = mpl.figure.Figure()
mpl.backends.backend_agg.FigureCanvasAgg
this_fig = frame
type(frame)


    
    
#ax = fig.gca()


#width = int(np.floor(width))




#ax.text(0.0,0.0,"Test", fontsize=45)
#ax.axis('off')

      # draw the canvas, cache the renderer

image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
plt.imshow(image)

#%%
%matplotlib inline
plt.ion()


plot_frames(plot_frames(these_indices,df_records, frames_npz))




fig2 = plt.figure()
fig2.axes.append(this_ax)

plt.show()


fig = plt.figure(facecolor='white')

frames[0]

ROWS = 1
COLS = 4
for i,this_ax in enumerate([f.axes[0] for f in frames]):
    print(this_ax)
    fig = fig.axes.append(this_ax)
    fig
    new_sub = fig.add_subplot(ROWS,COLS,i+1)
    new_sub.axes = this_ax
    fig.add_artist()
    this_ax.plot()
    
    dir(this_ax)
    
#%%
# TURN OFF PLOTTING


#records = get_full_records(frames_npz, df_records, df_records.index[0:3])
#for r in records:
#    this = gen_one_record_frame(r,"")
#type(this)





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
