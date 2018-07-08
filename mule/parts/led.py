from parts.base import BasePart
import random
import logging
import time
import RPi.GPIO as GPIO

class sequential_LED_loop(BasePart):
    ''' asdf '''
    input_keys = ()
    output_keys = ('led_flags',)
    #output_keys = ()
    #last_flags = state['led_flags']
    #print(last_flags)
    
    

    def __init__(self,PIN_BLUE1,PIN_BLUE2,PIN_BLUE3, PIN_BLUE4  ):
        NUMBER_LED = 4
        self.lights_off = [False for i in range(NUMBER_LED)]
        self.lights_on = [True for i in range(NUMBER_LED)]
        
        self.PIN_BLUE1   = PIN_BLUE1
        self.PIN_BLUE2   = PIN_BLUE2
        self.PIN_BLUE3   = PIN_BLUE3 
        self.PIN_BLUE4   = PIN_BLUE4
        
        self.count = 0
        
    def start(self):
        ''' asdf '''
        #state['led_flags'] = self.lights_on
        GPIO.setwarnings(False)
        
        GPIO.cleanup()
        
        GPIO.setmode(GPIO.BCM)
        
        GPIO.setup(self.PIN_BLUE1,GPIO.OUT)
        GPIO.setup(self.PIN_BLUE2,GPIO.OUT)
        GPIO.setup(self.PIN_BLUE3,GPIO.OUT)
        GPIO.setup(self.PIN_BLUE4,GPIO.OUT)

    def transform(self, state):
        ''' asdf
        '''
        #coin_flips = [random.random() < 0.5 for i in range(5)]
        #print(coin_flips)
        #state['led_flags'] = coin_flips

        if self.count%5 == 0:
            state['led_flags'] = self.lights_off
        else:
            state['led_flags'] = self.lights_off
            state['led_flags'][self.count%5-1] = True

        print(state['led_flags'])
        print(self.count)            
        self.count += 1
        
        GPIO.output(self.PIN_BLUE1,  state['led_flags'][0])
        GPIO.output(self.PIN_BLUE2,  state['led_flags'][1])
        GPIO.output(self.PIN_BLUE3,  state['led_flags'][2])
        GPIO.output(self.PIN_BLUE4,  state['led_flags'][3])
        
     
        
        #print(self.count,self.count%5)
        
        
        
        #print(bool(random.getrandbits(5)))
        #print(random.sample([True, False],5))
        #if coin_flip: state['led_flags'] = self.lights_on
        #else:  state['led_flags'] = self.lights_off

    def stop(self):
        '''  '''
        GPIO.output(self.PIN_BLUE1,  self.lights_off[0])
        GPIO.output(self.PIN_BLUE2,  self.lights_off[1])
        GPIO.output(self.PIN_BLUE3,  self.lights_off[2])
        GPIO.output(self.PIN_BLUE4,  self.lights_off[3])
    
    
    
    

class random_onoff_LED_loop(BasePart):
    ''' asdf '''
    input_keys = ()
    output_keys = ('led_flags',)
    #output_keys = ()

    def __init__(self,PIN_BLUE1,PIN_BLUE2,PIN_BLUE3, PIN_BLUE4  ):
        NUMBER_LED = 4
        self.lights_off = [False for i in range(NUMBER_LED)]
        self.lights_on = [True for i in range(NUMBER_LED)]
        
        self.PIN_BLUE1   = PIN_BLUE1
        self.PIN_BLUE2   = PIN_BLUE2
        self.PIN_BLUE3   = PIN_BLUE3 
        self.PIN_BLUE4   = PIN_BLUE4
        
    def start(self):
        ''' asdf '''
        #state['led_flags'] = self.lights_on
        GPIO.setwarnings(False)
        
        GPIO.cleanup()
        
        GPIO.setmode(GPIO.BCM)
        
        GPIO.setup(self.PIN_BLUE1,GPIO.OUT)
        GPIO.setup(self.PIN_BLUE2,GPIO.OUT)
        GPIO.setup(self.PIN_BLUE3,GPIO.OUT)
        GPIO.setup(self.PIN_BLUE4,GPIO.OUT)
        
    def transform(self, state):
        ''' asdf
        '''
        last_flags = state['led_flags']
        
        coin_flips = [random.random() < 0.5 for i in range(5)]
        #print(coin_flips)
        state['led_flags'] = coin_flips
        
        GPIO.output(PIN_BLUE1,  state['led_flags'][0])
        GPIO.output(PIN_BLUE2,  state['led_flags'][1])
        GPIO.output(PIN_BLUE3,  state['led_flags'][2])
        GPIO.output(PIN_BLUE4,  state['led_flags'][3])
        
        #print(bool(random.getrandbits(5)))
        #print(random.sample([True, False],5))
        #if coin_flip: state['led_flags'] = self.lights_on
        #else:  state['led_flags'] = self.lights_off

    def stop(self):
        '''  '''
        GPIO.output(PIN_BLUE1,  self.lights_off[0])
        GPIO.output(PIN_BLUE2,  self.lights_off[1])
        GPIO.output(PIN_BLUE3,  self.lights_off[2])
        GPIO.output(PIN_BLUE4,  self.lights_off[3])
        