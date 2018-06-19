#!/usr/bin/env python3

# TODO: this main file should be very simple, load in a template/configuration and set things going.

import logging.config
import yaml
import os
import warnings

logging_config_file = 'logging_simple.yaml'
path_logging_conf = os.path.join(os.getcwd(), 'logging', 'configurations', logging_config_file)
assert os.path.exists(path_logging_conf)
log_config = yaml.load(open(path_logging_conf, 'r'))
logging.config.dictConfig(log_config)

logging.info(f"Logging brought to you by {logging_config_file}")

#import utilities.other_utilities as util
#
#with warnings.catch_warnings(): # Suppress warnings!
#    warnings.simplefilter("ignore")
#    # Disable logging messages from tf - matplotlib (
#    #TODO: Better way??
#    logging.getLogger("matplotlib").setLevel(logging.WARNING)
#

from vehicle import Vehicle
from parts.camera import WebCam
from parts.display import DisplayFeed


def drive():
    #Initialize car
    logging.info(f"Start your engines ...")

    mule = Vehicle()
    
    mule.add(WebCam(threaded=True))
    #mule.add(DisplayFeed('MuleView'))

    mule.start()

    mule.drive()

    mule.stop()


if __name__ == '__main__':
    drive()
