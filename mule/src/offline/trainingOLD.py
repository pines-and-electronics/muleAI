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


