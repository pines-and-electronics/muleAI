import RPi.GPIO as GPIO
import time
import logging

GPIO.setwarnings(False)

PIN_BLUE1   = 26
PIN_BLUE2   = 19
PIN_BLUE3   = 13 
PIN_BLUE4   = 6

PIN_RED     = 16
PIN_YELLOW  = 20
PIN_GREED   = 21

GPIO.cleanup()

GPIO.setmode(GPIO.BCM)

GPIO.setup(PIN_BLUE1,GPIO.OUT)
GPIO.setup(PIN_BLUE2,GPIO.OUT)
GPIO.setup(PIN_BLUE3,GPIO.OUT)
GPIO.setup(PIN_BLUE4,GPIO.OUT)

GPIO.setup(PIN_RED,GPIO.OUT)
GPIO.setup(PIN_YELLOW,GPIO.OUT)
GPIO.setup(PIN_GREED,GPIO.OUT)

hertz = 10 
interval = 1 / hertz

def cycle_blue():
    GPIO.output(PIN_RED,    False)
    GPIO.output(PIN_YELLOW, False)
    GPIO.output(PIN_GREED,  False)    
    while 1:
        logging.debug("Cycle")

        GPIO.output(PIN_BLUE1,  True)
        GPIO.output(PIN_BLUE2,  False)
        GPIO.output(PIN_BLUE3,  False)
        GPIO.output(PIN_BLUE4,  False)
    
        time.sleep(interval)
    
        GPIO.output(PIN_BLUE1,  False)
        GPIO.output(PIN_BLUE2,  True)
        GPIO.output(PIN_BLUE3,  False)
        GPIO.output(PIN_BLUE4,  False)
    
        time.sleep(interval)
    
        GPIO.output(PIN_BLUE1,  False)
        GPIO.output(PIN_BLUE2,  False)
        GPIO.output(PIN_BLUE3,  True)
        GPIO.output(PIN_BLUE4,  False)
    
        time.sleep(interval)
    
        GPIO.output(PIN_BLUE1,  False)
        GPIO.output(PIN_BLUE2,  False)
        GPIO.output(PIN_BLUE3,  False)
        GPIO.output(PIN_BLUE4,  True)
        
        time.sleep(interval)

def cycle_onoff():
    while 1:
       
        logging.debug("LED Testing")
    
        GPIO.output(PIN_BLUE1,  True)
        GPIO.output(PIN_BLUE2,  True)
        GPIO.output(PIN_BLUE3,  True)
        GPIO.output(PIN_BLUE4,  True)
        GPIO.output(PIN_RED,    True)
        GPIO.output(PIN_YELLOW, True)
        GPIO.output(PIN_GREED,  True)
        
        time.sleep(1)
    
        GPIO.output(PIN_BLUE1,  False)
        GPIO.output(PIN_BLUE2,  False)
        GPIO.output(PIN_BLUE3,  False)
        GPIO.output(PIN_BLUE4,  False)
        GPIO.output(PIN_RED,    False)
        GPIO.output(PIN_YELLOW, False)
        GPIO.output(PIN_GREED,  False)
        
        time.sleep(1)
        
        #GPIO.cleanup()

if __name__ == "__main__":
    cycle_blue()
    #cycle_onoff()

