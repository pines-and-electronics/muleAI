#!/usr/bin/env python3
import click
import logging.config
import yaml
import os
import sys
    
from utilities import configure as configutil
from vehicle import Vehicle

CONFIG_DIR = 'configurations'
LOGGING_DIR = 'logging'

# LOGGING_DIR_HARD_CODE = "~/muleAI/mule/logging/logging.simple.yml"


@click.group()
@click.option('--logcfg', default=None, type=click.Path(exists=True))
# @click.option('--logcfg', default=LOGGING_DIR_HARD_CODE,type=click.Path())
# @click.option('--logcfg', default=None,type=click.Path(exists=True))
def cli(logcfg):
    print('Opening {}'.format(logcfg))
    with open(logcfg, 'r') as fd:
        config = yaml.load(fd)
        logging.config.dictConfig(config)

    logging.info('Logging brought to you by {}'.format(logcfg))


@cli.command()
def todo_create():
    pass


@cli.command()
def TODO_find():
    pass


@click.command()
@click.option('--cfg', default=os.path.join(CONFIG_DIR,'config.calibrate.yml'), type=click.Path(exists=True))
def todo_calibrate(cfg):
    pass




@click.command()
@click.option('--cfg', default=os.path.join(CONFIG_DIR,'config.drive.yml'), type=click.Path(exists=True))
@click.option('--model_path', type=click.Path(exists=True),required=False)
#@click.option('--logcfg', default=os.path.join(LOGGING_DIR, 'logging.simple.yml'), type=click.Path(exists=True))
def drive(cfg, model_path):

    config = configutil.parse_config(cfg)

    #TODO: Refactor passing model path cleanly!
    if model_path:
        logging.debug("Model: {}".format(model_path))
        for part in config['parts']:
            #print(part)
            for key in part.keys():
                if key=='ai':
                    part['ai']['arguments']['model_path'] = model_path
    else:
        logging.debug("No model specified.".format())

    #print(config)
    #raise
    protoparts = configutil.create_protoparts(config['parts'])
    
    logging.info('Creating vehicle from loaded configuration')

    mule = Vehicle.from_config(protoparts)

    logging.info('Start your engines ...')

    mule.start(config)

    logging.info('Initiating drive loop')

    mule.drive(freq_hertz=config['drive']['freq_hertz'],
               verbose=config['drive']['verbose'],
               verbosity=config['drive']['verbosity'],
               )

    logging.info('Killing engine')

    mule.stop()
    logging.info("Done with this driving session, exiting python.")


@cli.command()
def todo_train():
    pass


cli.add_command(drive)
cli.add_command(todo_calibrate)


if __name__ == '__main__':
    print('*** Welcome to Mule.AI ***')
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print("Current working directory", os.getcwd())
    cli()
