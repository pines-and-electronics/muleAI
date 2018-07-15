# -*- coding: utf-8 -*-
import os
DATA_PATH = r"/home/batman/TEMP_DEL"
DATA_ZIP = os.path.join(DATA_PATH , 'state.zip')
assert os.path.exists(DATA_ZIP)
                    
#%%

X_keys = ['cam/image_array']
y_keys = ['user/angle', 'user/throttle']


kl = KerasLinear()
