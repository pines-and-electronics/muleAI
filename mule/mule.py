#!/usr/bin/env python3

import logging.config
import yaml
import os

from utilities import configure as configutil
from vehicle import Vehicle



logging_config_file = 'logging_simple.yaml'
path_logging_conf = os.path.join(os.getcwd(), 'logging', 'configurations', logging_config_file)
assert os.path.exists(path_logging_conf)
log_config = yaml.load(open(path_logging_conf, 'r'))
logging.config.dictConfig(log_config)

logging.info('Logging brought to you by {}'.format(logging_config_file))



def drive(config):

    logging.info('Creating vehicle from config')

    mule = Vehicle.from_config(config.parts)

    logging.info('Start your engines ...')

    mule.start()

    logging.info('Initiating drive loop')

    mule.drive(**config.drive)

    logging.info('Killing engine')

    mule.stop()


if __name__ == '__main__':
    path = 'configurations/config.yml'
    config = configutil.parse_config(path)
    drive(config)
