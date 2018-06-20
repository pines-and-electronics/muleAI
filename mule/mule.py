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
from parts.camera import MockCam, PiCam, WebCam
from parts.datastore import ReadStore, WriteStore
from parts.display import DisplayFeed
from parts.joystick import PS3Controller
from parts.actuator import MockController, PCA9685Controller
from parts.actuator import SteeringController, ThrottleController
#from parts.ai import AIController
from parts.mock import MockMode


def drive():
    #Initialize car
    logging.info(f"Start your engines ...")

    mule = Vehicle()

    #mule.add(PiCam(threaded=True))
    mule.add(WebCam(threaded=True))

    mule.add(MockMode())

    #mule.add(ReadStore('../../datastores/1528813253', output_keys=('camera_array',)))

    # For the pc development
    mule.add(DisplayFeed('MuleView'))

    #mule.add(PS3Controller())

    #mule.add(AIController('nothing'))

    #mule.add(SteeringController(PCA9685Controller(channel=0)))
    #mule.add(ThrottleController(PCA9685Controller(channel=1)))

    mule.add(WriteStore('../../datastores'))

    mule.start()

    mule.drive()

    mule.stop()


if __name__ == '__main__':
    drive()
