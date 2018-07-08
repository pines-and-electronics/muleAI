#!/usr/bin/env python3
import click
import logging.config
import yaml
import os

from utilities import configure as configutil
from vehicle import Vehicle

CONFIG_DIR = 'configurations'
LOGGING_DIR = 'logging'

@click.group()
@click.option('--logcfg', default=os.path.join(LOGGING_DIR, 'logging.simple.yml'), type=click.Path(exists=True))
def cli(logcfg):
    with open(logcfg, 'r') as fd:
        config = yaml.load(fd)
        logging.config.dictConfig(config)

    logging.info('Logging brought to you by {}'.format(logcfg))


@cli.command()
def create():
    pass


@cli.command()
def find():
    pass


@click.command()
@click.option('--cfg', default=os.path.join(CONFIG_DIR,'config.calibrate.yml'), type=click.Path(exists=True))
def calibrate(cfg):
    pass

cli.add_command(calibrate)


@click.command()
@click.option('--cfg', default=os.path.join(CONFIG_DIR,'config.drive.yml'), type=click.Path(exists=True))
def drive(cfg):

    config = configutil.parse_config(cfg)
    
    protoparts = configutil.create_protoparts(config['parts'])
    
    logging.info('Creating vehicle from loaded configuration')

    mule = Vehicle.from_config(protoparts)

    logging.info('Start your engines ...')

    mule.start()

    logging.info('Initiating drive loop')

    mule.drive(freq_hertz=config['drive']['freq_hertz'])

    logging.info('Killing engine')

    mule.stop()

cli.add_command(drive)


@cli.command()
def train():
    pass



if __name__ == '__main__':
    cli()
