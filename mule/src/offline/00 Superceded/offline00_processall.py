#%% PROCESS ALL DATASETS!

#this_data_dir = LOCAL_PROJECT_PATH

#df_datasets = df_checkfiles
def get_datasets(df_datasets):
    
    """
    Iterate over each dataset.
    For each, create a new dictionary containing information. 
    Create any new files that are missing. 
    
    """
    
    # Iterate over each dataset
    dataset_def_list = list()
    for i,this_ds in df_datasets.iterrows():
        logging.debug("***********************".format())
        logging.debug("Dataset: {}".format(i))
        
        
        
        dataset_def = dict()
        #dataset_def['this_dir'] = this_dir
        #print(i,this_ds)
        #continue
        #return
        #for i,this_ds in enumerate(df_datasets['this_dir']):
        
        #    return
        # Date time from directory name
        dataset_def.update(process_datetime(i))
        
        # Numpy camera arrays
        dataset_def.update(check_camera_zip(this_ds['this_dir']))
        
        # json state records
        dataset_def.update(process_json_records(this_ds['this_dir']))
            
        # JPG zip
        try:
            dataset_def.update(process_jpg_zip(this_ds['this_dir']))
        except:
            pass
        
        # Video
        
        # If the JPG zip doesn't exist, create it
        # THIS IS CURRENTLY DISABLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not this_ds['jpg_images.zip'] and False:
            # JPG images
            #dataset_def = create_jpgs(dataset_def)
    
            # Write the JPG's
            #if not dataset_def['num_jpgs']:
            raise Exception("UPDATE THIS")
            path_jpg = write_jpg(dataset_def)
                # Reload the jpg definition!
             #   dataset_def = process_jpgs(dataset_def)
            
            # Zip them
            path_zip = os.path.join(dataset_def['this_dir'],'jpg_images.zip')
            zip_jpgs(path_jpg,path_zip)
            delete_jpgs(path_jpg)
            
            #dataset_def = zip_jpgs(dataset_def)

            dataset_def = process_jpg_zip(dataset_def)
            
        # Time step analysis
        dataset_def.update(process_time_steps(dataset_def['camera_numpy_zip'], dataset_def['json_record_zip']))
        

        #assert dataset_def['flg_jpg_zip_exists'], "jpg zip does not exist, {}".format(dataset_def)
        
        assert dataset_def['num_jpgs'] == dataset_def['num_records']
        
        logging.debug("Dataset {}: recorded on {}".format(i, dataset_def['this_dt_nice']))
        
        # Summary of this dataset
        logging.debug("{} aligned records found over {:0.2f} minutes.".format(
                dataset_def['num_records'],
                dataset_def['elapsed_time_mins']
                ))

        logging.debug("Timestep analysis: {:0.0f} +/- {:0.0f} ms".format(
                      dataset_def['ts_deltas_mean'],
                      dataset_def['ts_deltas_std'],
                      ))
        
        # Append
        this_series = pd.Series(dataset_def)
        this_series.name  = i
        dataset_def_list.append(this_series)
    
    # Done
    this_df_datasets = pd.DataFrame(dataset_def_list)
    #this_df_datasets = this_df_datasets.sort_values(['this_dt']).reset_index(drop=True)        
    
    return  this_df_datasets

df_datasets_processed = get_datasets(df_checkfiles)
df_all_datasets = df_checkfiles.join(df_datasets_processed)

#%% Print table

print(tabulate(df_all_datasets.loc[:,['camera_size_MB','elapsed_time']],headers="keys"))
for c in df_all_datasets.columns:
    print(c)

#%% Select one dataset
def select_data(this_df):
    head_str = "{:<20}  {:<30} {:<12} {:<15} {:>30}"
    row_str =  "{:<20}  {:<30} {:<12} {:>15.0f} {:>20.1f}"
    fields = ('this_dt_nice','num_records','camera_size_MB', 'elapsed_time_mins')
    head = ['idx'] + [*fields]
    print(head_str.format(*head))
    for i,r in this_df.iterrows():
        this_row_str = [r[fld] for fld in fields]
        print(row_str.format(i,*this_row_str))
    
    #ds_idx = int(input("Select dataset number:") )
    #this_dataset = this_df.iloc[ds_idx]
    #return this_dataset
selected_data = select_data(df_all_datasets)# -*- coding: utf-8 -*-

