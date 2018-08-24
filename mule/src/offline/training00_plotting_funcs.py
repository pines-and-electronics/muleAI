#sel_indices = list()
#sel_indices += df_records[df_records['steering_signal'] > 0.9].sample(4)['timestamp'].tolist()

def plot_frames(sel_indices,df_records, frames_npz):
    """
    Plot a row ro
    """
    fig=plt.figure(figsize=[20,18],facecolor='white')
    ROWS = 1
    COLS = 4
    NUM_IMAGES = ROWS * COLS
    #sel_indices = df_records.sample(NUM_IMAGES)['timestamp'].values
    sel_frames, sel_steerings, sel_throttles, these_ts = get_n_records(df_records, frames_npz, sel_indices)
    
    for i,record in enumerate(zip(sel_frames, sel_steerings,sel_throttles,these_ts)):
        this_frame, this_steer, this_throttle,sel_ts = record
        
        steer_cat = ''.join(['|' if v else '-' for v in linear_bin(this_steer)])
        timestamp_string = sel_ts.strftime("%D %H:%M:%S.") + "{:.2}".format(str(sel_ts.microsecond))
        
        this_label = "{}\n{:0.2f} steering\n{:0.2f} throttle".format(timestamp_string,this_steer,this_throttle)
        y = fig.add_subplot(ROWS,COLS,i+1)
        y.imshow(this_frame)
        #plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        t = y.text(5,25,this_label,color='green',alpha=1)
        #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
        t.set_bbox(dict(facecolor='white', alpha=0.3,edgecolor='none'))
        y.text(80,105,steer_cat,fontsize=30,horizontalalignment='center',verticalalignment='center',color='green')
        #plt.title()
