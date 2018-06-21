#!/usr/bin/env python3

import logging.config
import yaml
import os
import parts.actuator

from utilities import configure as configutil
from vehicle import Vehicle


logging_config_file = 'logging_simple.yaml'
path_logging_conf = os.path.join(os.getcwd(), 'logging', 'configurations', logging_config_file)
assert os.path.exists(path_logging_conf)
log_config = yaml.load(open(path_logging_conf, 'r'))
logging.config.dictConfig(log_config)

logging.info('Logging brought to you by {}'.format(logging_config_file))

def calibrate():
    """
    Simple function to set PWM values
    
    Choose a channel, set PWM
    Keyboard interrupt to quit
    """

    logging.info("Calibration of the PCA9685, keyboard interrupt to quit")
    
    address=0x40
    frequency=60
    logging.info("PCA9685 address set to {} (0x{:X}) - NOT USED?".format(address,address))
    logging.info("PCA9685 frequency set to {}".format(frequency))
    
    channel = int(input("Enter a channel number: "))
    
    pca = parts.actuator.PCA9685Controller(channel=channel)
    
    print(pca)
    
    for i in range(100):
        pwm_value = int(input("Enter a PWM value on channel {} (0-1500):".format(channel)))
        pca.run(pwm_value)

if __name__ == '__main__':
    calibrate()
