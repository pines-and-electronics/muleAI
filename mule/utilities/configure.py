import logging
import yaml
import importlib
from collections import namedtuple
import pprint

# POD classes to facilitate configuration parsing
# holder for main vehicle drive configuration
#class Config:
#    pass
# holder for attributes from which a part may be later constructed
class ProtoPart:
    pass

def parse_config(path):
    ''' Parses configuration file 

        Argument

        path: str
            path to configuration yaml file

        Can deal with two keys:
            drive : dict of argument to vehicle drive function
            parts : list of dicts

        Each part is a 1-element dict whose key is the name of the part module
        and whose value is a 2-element dict.

        In turn this 2-element dict contains the class type of the part (as a string)
        and the argument to be passed to this class upon initialization.
        
        returns a dictionary
    '''

    with open(path, 'r') as fd:
        config = yaml.load(fd)
    
    logging.info('Loaded configuration yaml file into dict: {}'.format(path))
    
    #pprint.pprint(config)
    
    for part in config['parts']:
        #print(part)
        print("")
        
        part_module = list(part.keys())
        assert len(part_module) == 1
        part_module = part_module.pop()
        
        print("Part type:", part[part_module]['type'])
        print("Part module: {}.py".format(part_module))
        if 'arguments' in part[part_module].keys():
            print("Part arguments:", part[part_module]['arguments'])
        else:
            print("Part arguments:", None)
        #print("")
    print("")
    #parsed_config = Config()

    #parsed_config.drive = config['drive'] if config.get('drive') else {}
    
    return config

def create_protoparts(parts):
    """
    A validator for parts
    
    From a config dictionary, go through each part and ensure the module and
    class exist. Return a list of 'protoparts'. 
    """    
    protoparts = []
    #print(parts)
    logging.debug('Processing {} parts'.format(len(parts)))

    for component in parts:
        # each component is a 1-element dict
        # name = name of parts submodule
        # attributes = elements needed to construct part
        name, attributes = component.popitem()

        name = 'parts.{}'.format(name) 
        
        logging.info('Processing {} {}'.format(name, attributes))

        
        # submodule of parts module containing the actual part
        module = importlib.import_module(name)
        
        protopart = ProtoPart()

        # attributes is a 2-element dict
        # type = class type of part
        # arguments = initialization arguments
        try:
            # Get the class from the module
            protopart.type = getattr(module, attributes['type'])
            # Get the arguments
            protopart.arguments = attributes['arguments'] if attributes.get('arguments') else {}
        except KeyError:
            logging.info('Part type not present in config: skipping {}'.format(name))
            continue

        logging.debug("Created a protopart: {} {}".format(protopart.type,protopart.arguments))

        protoparts.append(protopart)
    
    logging.debug("Returning {} protoparts ready for instantiation".format(len(protoparts)))

    #parsed_config.parts = protoparts
    
    #logging.info('Parsed config {}'.format(parsed_config))

    return protoparts
