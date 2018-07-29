**muleAI** is a lightweight python library that facilitates research and development in autonomous mobility at RC-scale. 

Mule: the love-child issuing from a male donkey (a jack) and a mare. Reliable. Sure footed. Even tempered. Gets you from point A to B. 

**muleAI** was inspired by the [DonkeyCar](http://www.donkeycar.com/) project. **muleAI** is a systematic ground-up re-implementation of some core functionality with some priorities in mind:
* A project for more experienced developers
* Simplicity and consistency in modular design
* Clean, well-structured implementation conforming to standard software design principles

[Sample run](https://youtu.be/Jmw1rkYdi4Y)

## Release candidate 1.0

### Dependencies
Tensor flow 1.8 (includes keras as `tf.keras`)

### Features
1. Extended configuration YAML file 
   * As much as possible is exposed to configuration, allowing rapid changing of parameters during racing days
1. Command line interface using [click](http://click.pocoo.org/5/)
1. Modular part classes inherit from Abstract Base Class
   * Enforce proper interface for all new parts
   * Include default behaviors such as class strings
1. Extensive logging messages throughout the project, for faster debugging
1. New adjustment method for PS3 controller, DPad selects a value, and up/down to change the value allowing performance changes as the car is driving
1. Images are saved directly to numpy arrays

### Changes
1. No support for any installation or setup method - project is run directly from the git directory. 
1. Linux and Mac OS are tested, not Windows
1. Training of models is left up to the user, in a separate module

