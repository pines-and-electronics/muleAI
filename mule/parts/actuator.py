import math
import functools
import time
from parts.base import BasePart
import logging


# TODO: should we really have separate instances for the steering and throttle
#       on the same board
# TODO: check that _signal2pulse allows for the fact that the user may put 
#       full_left_pulse < full_right_pulse or
#       full_reverse_pulse > full_forward_pulse
# TODO: document the above idiosyncracy

def _signal2pulse(signal_lower, signal_upper, pulse_lower, pulse_upper, signal):
    ''' Linearly transforms signal from signal-space to pulse-space

        signal_lower, signal_upper: float
            limits (in principle) for signal values

        pulse_lower, pulse_upper: int
            limits for pulse values (discovered during callibration)

        signal: float
            value to be linearly transformed
    '''
    slope =  (pulse_upper - pulse_lower) / (signal_upper - signal_lower)
    pulse = slope * (signal - signal_lower) + pulse_lower

    return int(math.floor(pulse + 0.5))



class MockController:
    ''' A mock controller class with the appropriate methods '''

    def set_pulse(self, pulse):
        ''' Set pwm pulse '''
        pass



# TODO: fix documentation
class PCA9685Controller:
    ''' 
    PCA 9685 16 Channel 12-bit PWM Servo Shield Motor Driver I2C module

    self.pwm has three methods:

    * set_pwm_freq(frequency) sets frequency of pwm in Hz

    * set_pwm(channel, on, off)

    * set_pwm_all(on, off)
    '''
    input_keys = ()
    output_keys = ()
    
    def __init__(self, address=0x40, frequency=60, channel=0):
        ''' Create a reference to the PCA9685 on a specified channel '''

        import Adafruit_PCA9685
        logging.debug("Adafruit_PCA9685 library imported")
        print("Loggers:")
        print(logging.Logger.manager.loggerDict)
        
        logging.debug("Instantiating PCA9685 class at address 0x{:X}".format(address))
        self.PCA9685 = Adafruit_PCA9685.PCA9685()
        
        logging.debug("Set the PWM frequency {} Hz".format(frequency))
        self.PCA9685.set_pwm_freq(frequency)

        self.channel = channel

    def set_pulse(self, pulse):
        ''' Set pwm pulse '''
        self.PCA9685.set_pwm(self.channel, 0, pulse) 

    def run(self, pulse):
        self.set_pulse(pulse)

    def start(self):
        pass

    def transform(self, PWM_value):
        pass

    def stop(self):
        pass


class SteeringController(BasePart):
    ''' Controls vehicle steering '''

    input_keys = ('steering_signal',)
    output_keys = ()

    FULL_LEFT_SIGNAL = 1 
    FULL_RIGHT_SIGNAL = -1
    STRAIGHT_SIGNAL = 0

    def __init__(self, controller_select="Mock",
                    channel = 1,
                       full_left_pulse=490,
                       full_right_pulse=290):
        ''' Acquires reference to controller and full left and right pulse frequencies
            that are discovered during callibration '''

        if controller_select == "Mock":
            controller = MockController()
        elif controller_select == "PCA":
            controller = PCA9685Controller()
        else:
            raise

        
        self.controller = controller

        self._steering_signal2pulse = functools.partial(_signal2pulse, self.FULL_LEFT_SIGNAL, 
                                                                       self.FULL_RIGHT_SIGNAL, 
                                                                       full_left_pulse, 
                                                                       full_right_pulse)

    def start(self):
        pass

    def transform(self, state):
        ''' Send signal as pulse to servo '''
        pulse = self._steering_signal2pulse(state['steering_signal'])
        self.controller.set_pulse(pulse)

    def stop(self):
        ''' Signal the servo to return to straight trajectory '''
        pulse = self._steering_signal2pulse(self.STRAIGHT_SIGNAL)
        self.controller.set_pulse(pulse)

    @property
    def _class_string(self):
        return "{} with {} {} left/right PWM".format(self.__class__.__name__, self.FULL_LEFT_SIGNAL,self.FULL_RIGHT_SIGNAL)


class ThrottleController(BasePart):
    ''' Controls vehicle throttle '''

    input_keys = ('throttle_signal',)
    output_keys = ()

    FULL_REVERSE_SIGNAL = -1
    FULL_FORWARD_SIGNAL =  1
    NEUTRAL_SIGNAL = 0

    def __init__(self, controller_select="Mock",
                    channel = 0,
                       full_reverse_pulse=290,
                       full_forward_pulse=490,
                       neutral_pulse=390):
        ''' Acquires reference to controller and full forward and reverse as well 
            as pulse frequencies that are discovered during callibration 

            One needs both signal to pulse maps as the pulse-ranges for reverse and forward 
            need not be the same
            '''

        if controller_select == "Mock":
            controller = MockController()
        elif controller_select == "PCA":
            controller = PCA9685Controller()
        else:
            raise
        
        self.controller = controller

        self._reverse_signal2pulse = functools.partial(_signal2pulse, self.FULL_REVERSE_SIGNAL,
                                                                      self.NEUTRAL_SIGNAL,
                                                                      full_reverse_pulse,
                                                                      neutral_pulse)

        self._forward_signal2pulse = functools.partial(_signal2pulse, self.NEUTRAL_SIGNAL,
                                                                      self.FULL_FORWARD_SIGNAL,
                                                                      neutral_pulse,
                                                                      full_forward_pulse)


    def start(self):
        ''' Callibrate by sending stop signal '''
        pulse = self._reverse_signal2pulse(self.NEUTRAL_SIGNAL)
        self.controller.set_pulse(pulse)
        time.sleep(0.5)


    def transform(self, state):
        ''' Send signal as pulse to servo '''
        if state['throttle_signal'] > self.NEUTRAL_SIGNAL:
            pulse = self._forward_signal2pulse(state['throttle_signal'])
        else:
            pulse = self._reverse_signal2pulse(state['throttle_signal'])

        self.controller.set_pulse(pulse)


    def stop(self):
        ''' Signal to stop vehicle ''' 
        pulse = self._reverse_signal2pulse(self.NEUTRAL_SIGNAL)
        self.controller.set_pulse(pulse)
        
    @property
    def _class_string(self):
        return "{} with".format(self.__class__.__name__)

        
