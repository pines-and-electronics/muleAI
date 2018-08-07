
from tensorflow.python.keras.models import load_model
from parts.base import BasePart

import numpy as np

# TODO: add assertions for keras and tensorflow
# TODO: add consistency checking between model and inputs??
#       or at least error handling
# TODO: could this be multiprocessed? or would this interfere with the
#       drive loop?
# TODO: do we need adapter components to wrap the keras model --> this could
#       already be assigned to the image processing handler
# TODO: At the moment, the model is tied to keras. Is this a good thing?


class AIController(BasePart):
    ''' AI that generates steering and throttle signals '''

    input_keys = ('mode', 'camera_array',)
    output_keys = ('steering_signal', 'throttle_signal')

    def __init__(self, model_path, input_keys=None, output_keys=None):
        ''' Loads controlling ai model

            Arguments

            model_path: str
                path to ai model
        '''
        self.input_keys = input_keys or self.input_keys
        self.output_keys = output_keys or self.output_keys
        self.model = load_model(model_path)
        self.model_path = model_path

    def start(self):
        pass


    def transform(self, state):
        ''' Updates state with output of ai model '''
        if state['mode']['steering'] == 'ai':
            this_img = state['camera_array']
            this_img = np.expand_dims(this_img, 0)
            steering_prediction = self.model.predict(this_img)
            steering_prediction = steering_prediction*5
            if steering_prediction > 1: 
                steering_prediction = 1
            elif steering_prediction < -1:
                steering_prediction = -1
                
            state['steering_signal'] = steering_prediction
            
            print(state['steering_signal'])

#        if state['mode'].steering == 'human' and state['mode'].throttle == 'human':
#            pass
#        else:
#            steering_signal, throttle_signal = self.model.predict(state['camera_array'])
#
#            if state['mode']['steering'] == 'ai':
#                state['steering_signal'] = steering_signal
#
#            if state['mode']['throttle'] == 'ai':
#                state['throttle_signal'] = throttle_signal


    def stop(self):
        pass
    
    @property
    def _class_string(self):
        return "{} from {}".format(self.__class__.__name__, self.model_path)

    
