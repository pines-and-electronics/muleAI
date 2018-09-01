Mule: the love-child issuing from a male donkey (a jack) and a mare. Reliable. Sure footed. Even tempered. Gets you from point A to B, even if it's not the most elegant ride in town. 

---

**muleAI** is a lightweight python library that facilitates research and development in autonomous mobility at RC-scale. 

**muleAI** is a foundation for further experiments in mobility, autonomous hardware, embedded AI, Internet Of Things, ...

**muleAI** was inspired by the [DonkeyCar](http://www.donkeycar.com/) project. **muleAI** is a systematic ground-up re-implementation of some core functionality with some priorities in mind:
* A project for more experienced developers
* Simplicity and consistency in modular design
* Clean, well-structured implementation conforming to standard software design principles

[Sample run](https://youtu.be/Jmw1rkYdi4Y)

## Release candidate 1.0

### Dependencies
UPDATE? Tensor flow 1.8 (includes keras as `tf.keras`)

### Features
1. Extended configuration YAML file 
   * As much as possible is exposed to configuration, allowing rapid changing of parameters during racing days
1. Command line interface using [click](http://click.pocoo.org/5/)
1. Modular part classes inherit from Abstract Base Class
   * Enforce proper interface for all new parts
   * Include default behaviors such as class strings
1. Extensive logging messages throughout the project, for faster debugging
1. New adjustment method for PS3 controller, DPad selects a value, and up/down to change the value allowing performance changes as the car is driving
   * D-pad left/right on the PS3 controller iterates over adjustment settings
   * D-pad up/down on the PS3 controller adjusts that value by SHIFT amount
   * Currently able to adjust max forward/reverse throttle and steering
1. Images are saved directly to numpy arrays, timestamped, and zipped for fast transfer to training
1. After driving, analysis and training of results are done in an offline suite of tools
   * DataSet class 
    * Enforces contract for further offline processing pipeline
    * Strictly aggregates the numpy images and the saved records and signals
    * Transforms underlying records
   * Plotter class
    * Operates on DataSet to plot summary histograms, charts etc.
    * Operates on DataSet to generate analysis frames with a HUD overlay
   * DataGenerator class operates on DataSet to serve batches to Keras
   * Trainer class operatates on DataSet to train a Keras model
   * Saliency class operates on DataSet to generate


### Behaviour changes
1. No support for any installation or setup method - project is run directly from the git directory. 
1. Linux and Mac OS are tested, not Windows
1. Training of models is in a separate module

