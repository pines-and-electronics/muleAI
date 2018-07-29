import time

from itertools import count
from utilities.generic import regulate
import logging
#from numpy.core.multiarray_tests import npy_log10l

import numpy as np
class Vehicle():
    ''' Vehicle control

        Methods to:

        * add part to vehicle
        * start vehicle, starting all parts
        * engage drive loop
        * stop vehicle, stopping all parts
    '''

    def __init__(self):
        ''' Members

            self.state_keys: set
                register of keys used as inputs and outputs for 
                different vehicle parts

            self.state: dict
                keys are ultimately all those self.state_keys registered in by self.add
                values are the current state values, e.g.
                    key = 'camera_image_array'
                    value = numpy.array containing rgb data from camera feed

            self.parts: list
                sequence of parts that transform self.state
        '''
        self.state_keys = set()

        self.state = {}
        self.parts = []
        logging.info("Initialize Vehicle")

    def print_state(self):
        """Helper to print concise state"""
        state_strings = list()
        #print(self.state)
        for key in self.state:
            
            #print(key)
            #print())
            #if self.state[key] == 
            #print(self.state[key])
            if type(self.state[key]) == np.ndarray:
                this_variable = self.state[key].shape
            else:
                this_variable = self.state[key]
            state_strings.append("{}={}".format(key,this_variable))
        #raise
    
        return ", ".join(state_strings)
     
    def add(self, part):
        ''' Adds part to vehicle

            Arguments

            part: parts.BasePart
            parts.BasePart is an ABC for a generic vehicle part
        '''
        logging.info("Adding part {}, {}".format(len(self.parts)+1,part))

        if not self.state_keys.issuperset(set(part.input_keys)):
            msg='state missing input key for {}'
            raise KeyError(msg.format(part.__class__.__name__))

        self.parts.append(part)

        logging.debug("Registering {} output key(s)".format(len(part.output_keys)))

        self.state_keys = self.state_keys.union(set(part.output_keys))
        
        logging.debug("Keys: {}".format(self.state_keys))


    @classmethod
    def from_config(cls, config):
        ''' Creates vehicle from parsed configuration

            Argument

            config: list(utilities.ProtoParts)
                ProtoPart is a POD class; it is expected to
                have members:
                    type : class type for vehicle part
                    arguments: dict of kwargs to be passed to initializer
        '''
        logging.info('Loading parts from parsed config')

        vehicle = cls()

        for part in config:
            logging.debug("Adding {} {}".format(part.type, part.arguments))
            

            vehicle.add(part.type(**part.arguments))

        return vehicle




    def start(self,config):
        ''' Starts vehicle by starting it constituent parts '''
        self.state = dict.fromkeys(self.state_keys, None)


        for i, part in enumerate(self.parts):
            logging.info("Starting part {} of {}, {}".format(i+1,len(self.parts),part))
            part.start()


    # TODO: log if steps per second are unattainable
    # TODO: log moving average of loop times
    # TODO: implement maximum number of drive loops ???  
    #       I fail to see the usefulness at the moment
    def drive(self, freq_hertz=10):
        ''' Engages drive loop

            Arguments

            freq_hertz: int
                number of drive loops to complete per second

            Iterates through parts, each transforming the state. Contains
            regulator that ensures freq_hertz.
        '''

        logging.info("Starting drive loop at {} Hz".format(freq_hertz))

        LOOP_VERBOSE = False
        LOOP_VERBOSITY = 20
        try:
            for loop_nr in regulate(count(), freq_hertz):
                # For debugging: Print current status
                if LOOP_VERBOSE and loop_nr%LOOP_VERBOSITY == 0:
                    print("Loop",loop_nr, "Vehicle state variables: ",self.print_state())
                    
                for part in self.parts:
                    part.transform(self.state)
                    
                    # For debugging: print each part
                    if LOOP_VERBOSE and loop_nr%LOOP_VERBOSITY == 0:
                            print("\t",part)

        # TODO: log detection of keyboard interrupt to screen
        #       and notify that this is expected behaviour
        except KeyboardInterrupt:
            pass


    def stop(self):
        ''' Stops vehicle by stopping parts in reverse order ''' 
        for part in reversed(self.parts):
            logging.debug("Stopping {} ...".format(part._class_string)) 
            part.stop()
