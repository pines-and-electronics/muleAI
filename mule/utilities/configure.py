import logging
import yaml
import importlib
from collections import namedtuple

# POD classes to facilitate configuration parsing
# holder for main vehicle drive configuration
class Config:
    pass
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
    '''
    logging.info('Parsing configuration yaml file: {}'.format(path))

    with open(path, 'r') as fd:
        config = yaml.load(fd)


    parsed_config = Config()

    parsed_config.drive = config['drive'] if config.get('drive') else {}

    protoparts = []

    for component in config.get('parts', []):
        # each component is a 1-element dict
        # name = name of parts submodule
        # attributes = elements needed to construct part
        name, attributes = component.popitem()

        name = 'parts.{}'.format(name) 

        # submodule of parts module containing the actual part
        module = importlib.import_module(name)

        protopart = ProtoPart()

        # attributes is a 2-element dict
        # type = class type of part
        # arguments = initialization arguments
        try:
            protopart.type = getattr(module, attributes['type'])
            protopart.arguments = attributes['arguments'] if attributes.get('arguments') else {}
        except KeyError:
            logging.info('Part type not present in config: skipping {}'.format(name))
            continue

        logging.debug(protopart.type)

        protoparts.append(protopart)

    parsed_config.parts = protoparts

    return parsed_config
