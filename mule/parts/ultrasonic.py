import RPi.GPIO as GPIO
import time


import logging

PIN_TRIG = 23 
PIN_ECHO = 24

while 1:
    
    GPIO.setmode(GPIO.BCM)
    
    print("Distance Measurement In Progress")
    logging.debug("test")
    
    GPIO.setup(PIN_TRIG,GPIO.OUT)
    GPIO.setup(PIN_ECHO,GPIO.IN)
    
    GPIO.output(PIN_TRIG, False)
    
    print("Waiting For Sensor To Settle... 1")
    
    time.sleep(1)
    
    print("Waiting For Sensor To Settle... 2")
    
    time.sleep(1)
    
    # Pulse TRIG
    GPIO.output(PIN_TRIG, True)
    time.sleep(0.00001)
    GPIO.output(PIN_TRIG, False)
    
    # Wait for start of signal
    while GPIO.input(PIN_ECHO)==0:
        pulse_start = time.time()
    
    # Wait for end of signal
    while GPIO.input(PIN_ECHO)==1:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start
    
    logging.debug("Pulse duration: {}".format(pulse_duration))

    distance = pulse_duration * 17150
    
    distance = round(distance, 2)
    
    print("Distance:",distance,"cm")
    
    GPIO.cleanup()