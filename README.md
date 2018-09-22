**The Mule**: the forbidden love-child issuing from a male donkey (a jack) and a mare. Sure footed. Even tempered. Ok, maybe your friends laugh at you from their fancy horses, their thoroughbreds, their imported Arabians; but we'll see who gets the last laugh!

---

**muleAI** is inspired by the [DonkeyCar](http://www.donkeycar.com/) project. We decided not to fork, but to rewrite. 

**muleAI** is a systematic re-implementation of some core functionality with some priorities in mind:
* Simplicity and consistency in modular design
* Clean, well-structured implementation conforming to standard software design principles

**muleAI** is a lightweight python library that facilitates research and development in autonomous mobility at RC-scale. 

**muleAI** is a foundation for further experiments in mobility, autonomous hardware, embedded AI, Internet Of Things, ...

## Release candidate 1.0

### Dependencies
UPDATE? ~~Tensor flow 1.8 (includes keras as `tf.keras`)~~

### Features - `Mule` autonomous vehicle operations software platform
1. Extended configuration file, YAML format
   * As much as possible is exposed to configuration, allowing rapid changing of parameters during racing days
1. Command line interface exposed using [click](http://click.pocoo.org/5/)
1. States are saved using a timestamp, the `time.time() * 1000` (Unix standard, number of milliseconds since 1970)
   * Allows for fast timestep analysis, strict ordering and alignment of states
1. Modular part classes inherit from Abstract Base Class
   * Enforce proper interface for all new parts
   * Include default behaviors such as class strings
1. Extensive logging messages throughout the project, for faster debugging
1. New adjustment method for PS3 controller, DPad selects a value, and up/down to change the value allowing performance changes as the car is driving
   * D-pad left/right on the PS3 controller iterates over adjustment settings
   * D-pad up/down on the PS3 controller adjusts that value by SHIFT amount
   * Currently able to adjust max forward/reverse throttle and steering
1. Images are saved directly to numpy arrays, timestamped, and zipped for fast transfer to training

### Behaviour changes
1. No support for any installation or setup method - project is run directly from the git directory
1. Linux and Mac OS are tested, not Windows
1. Training of models is in a separate module


