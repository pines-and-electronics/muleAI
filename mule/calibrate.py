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

    logging.info('Calibration of the PCA9685')
    
    address=0x40
    frequency=60
    logging.info("PCA9685 address set to {} (0x{:X}) - NOT USED?".format(address,address))
    logging.info("PCA9685 frequency set to {}".format(frequency))
    
    channel = int(input('Enter a channel number: '))
    
    pca = parts.actuator.PCA9685Controller(channel=channel)
    
    print(pca)
    
    for i in range(10):
        pwm_value = int(input('Enter a PWM setting to test (0-1500): '))
        pca.run(pwm_value)
        #c.run(pmw)
    
    
    # Get the actuator back
    #print(mule.parts)
    
    #logging.info('Start your engines ...')

    #mule.start()

    #logging.info('Initiating drive loop')

    #mule.drive(**config.drive)

    #logging.info('Killing engine')

    #mule.stop()

if __name__ == '__main__':
    #path = 'configurations/config_calibration.yml'
    #config = configutil.parse_config(path)
    calibrate()
