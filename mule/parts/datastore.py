import os.path
import time
import re
import json
import numpy as np
import random
import itertools
from collections import OrderedDict
from parts.base import BasePart
import datetime
#from samba.dcerpc.drsblobs import ldapControlDirSyncBlob
import logging
#import zipfile
import shutil

# TODO: add handler for multiple read/write stores
# TODO: document properly
# TODO: multithread, multiprocess or asyncio these ops


def _json_load(path):
    ''' Wrapper to load contents of json file '''
    with open(path, 'r') as fd:
         return json.load(fd)



class ReadStore(BasePart):
    ''' Reads data from storage path into a state dictionary

        Expects filenames in a certain format:

        <state-key>_<timestamp-in-seconds>.<extension>
    '''
    input_keys = ()
    output_keys = ()

    def __init__(self, path, output_keys, shuffled=False):
        ''' Retrieve file details and set up state generator

            Arguments

            path: str
                path to datastore

            output_keys: tuple
                keys for state to retrieve

            shuffled: bool
                toggle return of state sequence in shuffled
                or sorted manner


            Members

            specifiers: collections.OrderedDict
                file specifiers extracted from filename
                key = timestamp
                value = list of tuples (label, extension, filename)

                Either label is a stete key for a file containing a
                numpy array or label = 'state', meaning the (json) file
                contains state data

            state_generator: generator
                yields next state from file
        '''
        self.path = path
        self.output_keys = output_keys
        self.specifiers = self._retrieve_specifiers(shuffled)
        self.state_generator = self._update()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = os.path.realpath(os.path.expanduser(path))

        # TODO: remove to external verification
        if not os.path.exists(self.path):
            msg = '{} does not exist'.format(self.path)
            raise FileNotFoundError()


    def start(self):
        pass

    def transform(self, state):
        ''' Generate state from file contents and assign
            vehicle state

            Notes:
                The explicit iteration through the private
                self._state is required because assigning directly
                to state would hide it from the enclosing scope
        '''
        self._state = next(self.state_generator)
        for key, value in self._state.items():
            state[key] = value


    def stop(self):
        pass


    def _retrieve_specifiers(self, shuffled):
        ''' Extract file information from filenames

            Arguments

            shuffled: bool
                shuffle or sort OrderedDict


            Returns

            OrderedDict containing file metadata
                key = timestamp
                value = (label, extension, filename)

                Either label is a stete key for a file containing a
                numpy array or label = 'state', meaning the (json) file
                contains state data
        '''
        # filename pattern
        # (label)_(timestamp).(extension)
        PATTERN = '(\w+)_([0-9]*).(.*)'

        specifiers = {}
        for node in os.listdir(self.path):
            match = re.fullmatch(PATTERN, node)
            if match:
                filename = os.path.join(self.path, node)
                label, timestamp, extension = match.groups()
                timestamp = int(timestamp)

                #
                specifier = (label, extension, filename)
                if specifiers.get(timestamp):
                    specifiers[timestamp].append(specifier)
                else:
                    specifiers[timestamp] = [specifier]

        # return data shuffled or sorted
        timestamps = list(specifiers)
        if shuffled:
            random_generator = random.Random()
            random_generator.seed(0)
            random_generator.shuffle(timestamps)
        else:
            timestamps.sort(key=lambda t: int(t))

        return OrderedDict((timestamp, specifiers[timestamp])
                           for timestamp in timestamps)



    def _update(self):
        ''' Generates state dictionary from files

            Expects fiies to contain numpy arrays or json metadata'''
        # contruct state from different data types
        state = {}
        for timestamp, specifier in self.specifiers.items():
            for label, extension, filename in specifier:
                if extension == 'npy' and label in self.output_keys:
                    state[label] = np.load(filename)
                elif extension == 'json':
                    data = _json_load(filename)
                    for key, value in data.items():
                        if key in self.output_keys:
                            state[key] = value
                else:
                    raise ValueError('Invalid extension: {}'.format(extension))
            yield state



class WriteStore(BasePart):
    ''' Writes state dictionary to storage path

        Writes filenames in a certain format:

        <state-key>_<timestamp-in-seconds>.<extension>
    '''
    input_keys = ('mode',)
    output_keys = ()

    def __init__(self, path, flg_zip):
        self.path = path
        self.flg_zip = flg_zip

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = os.path.realpath(os.path.expanduser(path))
        date_time_str = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
        self._path = os.path.join(self._path, date_time_str)

        # Ensure the path is new
        assert not(os.path.exists(self._path))

        # Create the folder
        # TODO: remove to external verification - Why? -MJ
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        logging.debug("Session data folder created: {}".format(self._path))


    def make_archive(self, source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        #print(source, destination, archive_from, archive_to)
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)
        
        logging.debug("Created archive {}".format(destination))
        
    def start(self):
        pass


    def transform(self, state):
        ''' Writes state to (multiple) files

            Writer first disposes of all the numpy arrays and then
            writes remaining data to json file
        '''
        #print("Transform. Mode:", state['mode'])
        if state['mode']['recording']:
            # time in milliseconds
            timestamp = int(time.time() * 1e3)

            # convert to set to remove duplicates
            keys = (key for key in state if key not in self.input_keys)
            self.input_keys = tuple(key for key
                                    in itertools.chain(self.input_keys, keys))

            local_state = {}

            for key in self.input_keys:
                if isinstance(state[key], np.ndarray):
                    filename = os.path.join(self.path,
                            '{}_{}.{}'.format(key, timestamp, 'npy'))
                    np.save(filename, state[key])

                else:
                    local_state[key] = state[key]

            filename = os.path.join(self.path, 'state_{}.{}'.format(timestamp, 'json'))
            with open(filename, 'w') as fd:
                json.dump(local_state, fd)


    def stop(self):

        # Get statistics of the saved data
        files = os.listdir(self.path)
        npy_files = [f for f in files if os.path.splitext(f)[1]=='.npy']
        json_files = [f for f in files if os.path.splitext(f)[1]=='.json']
        other_files = [f for f in files if os.path.splitext(f)[1] not in ('.npy','.json')]
        
        assert len(npy_files) == len(json_files)
        
        logging.debug("{} states (.npy, .json pairs)saved to {}".format(len(json_files),self.path))
        logging.debug("{} other files saved to {}".format(len(other_files),self.path))
        
        self.make_archive(self.path,os.path.join(self.path,'state.zip'))
        
        # Remove all .npy files, confirm
        [os.remove(os.path.join(self.path,f)) for f in npy_files]
        #files = os.listdir(self.path)
        npy_files = [f for f in os.listdir(self.path) if os.path.splitext(f)[1]=='.npy']
        assert len(npy_files) == 0
        logging.debug("Deleted all .npy files".format())
        
        # Remove all .json files, confirm
        [os.remove(os.path.join(self.path,f)) for f in json_files]
        #files = 
        json_files = [f for f in os.listdir(self.path) if os.path.splitext(f)[1]=='.json']
        assert len(json_files) == 0
        logging.debug("Deleted all .json files".format())        