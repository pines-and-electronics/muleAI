'''
    The structure of JoystickDevice takes heavy inspiration from:
    https://gist.github.com/rdb/8864666
    released by rdb under the Unlicense

    There are a number of inaccuracies in the above implementation, to which
    the linux joystick api appears to be robust:
    e.g. JSIOCGBTNMAP == 0x84006a34 rather than 0x80406a34
         MAX_NR_BUTTONS == 0x200 == 512 rather than 200

    If you wish to obtain the values of JSIOCG[NAME/AXES/AXMAP/BUTTONS/BTNMAP]
    on your system, you will need to evaluate the appropriate quantities from
    your system header: linux/joystick.h
    or you can use those provided by the included utilities.jsio
'''
import fcntl
import array
import struct
#from collections import OrderedDict
import collections
from parts.base import BasePart, ThreadComponent
import utilities.jsio as jsio
import pprint
import logging


from bitstring import BitArray


TESTING = True

# TODO: potentially migrate from the older joystick interface 
#       to the evdev interface

# TODO: include testing kit in repo
# TODO: make this initial configuration better.
''' * Only Dualshock 3 keys mapped so far
    * Axes and buttons need different lookup tables as they have in principle 
      overlapping ranges of keys:
      https://www.kernel.org/doc/html/v4.12/input/joydev/joystick-api.html#js-event-number
    * Adding key mappings for your own joystick is currently a multi-step procedure:
        1. sudo apt-get joystick
        2. jstest /dev/input/js<number>
        3. jstest presents two arrays: axis array and button array. By activating 
           the various buttons/axes, record (manually) the order associated
           to the axes/buttons in their respective arrays (0,1,2,...).  
        4. You need to run the fcntl.ioctl functions within JoystickDevice to 
           retrieve JSIOCGAXMAP and JSIOCGBTNMAP.  This results in two further arrays.
           Using the order information from jstest, one can map physical axes/buttons
           to keys sent.
'''
_AXIS_NAME_LOOKUP_OLD = {                   # DEC   | linux jsio naming
        0x00: 'axis-thumb-left-x',      # 0     | 'x'
        0x01: 'axis-thumb-left-y',      # 1     | 'y'
        0x02: 'axis-thumb-right-x',     # 2     | 'z'
        0x05: 'axis-thumb-right-y',     # 5     | 'rz' (rotate-z)
        0x28: 'axis-?-1',               # 40    |
        0x29: 'axis-?-2',               # 41    |
        0x2a: 'axis-?-3',               # 42    |
        0x2b: 'axis-?-4',               # 43    |
        0x2c: 'axis-dpad-up',           # 44    |
        0x2d: 'axis-dpad-right',        # 45    |
        0x2e: 'axis-dpad-down',         # 46    |
        0x2f: 'axis-dpad-left',         # 47    |
        0x30: 'axis-trigger-left-2',    # 48    |
        0x31: 'axis-trigger-right-2',   # 49    |
        0x32: 'axis-trigger-left-1',    # 50    |
        0x33: 'axis-trigger-right-1',   # 51    |
        0x34: 'axis-triangle',          # 52    |
        0x35: 'axis-circle',            # 53    |
        0x36: 'axis-cross',             # 54    |
        0x37: 'axis-square',            # 55    |
        0x38: 'axis-?-5',               # 56    |
        0x39: 'axis-?-6',               # 57    |
        0x3a: 'axis-?-7',               # 58    |
        0x3b: 'axis-accelerometer-x',   # 59    |
        0x3c: 'axis-accelerometer-y',   # 60    |
        0x3d: 'axis-accelerometer-z',   # 61    |
        0x3e: 'axis-?-8'                # 62    |
        }


_AXIS_NAME_LOOKUP = {
0x00 : 'x',        
0x01 : 'y',        
0x02 : 'z',        
0x03 : 'rx',        
0x04 : 'ry',        
0x05 : 'rz',        
0x06 : 'trottle',        
0x07 : 'rudder',        
0x08 : 'wheel',        
0x09 : 'gas',        
0x0a : 'brake',        
0x10 : 'hat0x',        
0x11 : 'hat0y',        
0x12 : 'hat1x',        
0x13 : 'hat1y',        
0x14 : 'hat2x',        
0x15 : 'hat2y',        
0x16 : 'hat3x',        
0x17 : 'hat3y',        
0x18 : 'pressure',        
0x19 : 'distance',        
0x1a : 'tilt_x',        
0x1b : 'tilt_y',        
0x1c : 'tool_width',        
0x20 : 'volume',        
0x28 : 'misc',    
        }





_BUTTON_NAME_LOOKUP_OLD = {
        0x120: 'button-select',         # 288   | 'trigger'
        0x121: 'button-thumb-left',     # 289   | 'thumb'
        0x122: 'button-thumb-right',    # 290   | 'thumb2'
        0x123: 'button-start',          # 291   | 'top'
        0x124: 'button-dpad-up',        # 292   | 'top2'
        0x125: 'button-dpad-right',     # 293   | 'pinkie'
        0x126: 'button-dpad-down',      # 294   | 'base'
        0x127: 'button-dpad-left',      # 295   | 'base2'
        0x128: 'button-trigger-left-2', # 296   | 'base3'
        0x129: 'button-trigger-right-2',# 297   | 'base4'
        0x12a: 'button-trigger-left-1', # 298   | 'base5'
        0x12b: 'button-trigger-right-1',# 299   | 'base6'
        0x12c: 'button-triangle',       # 300   | 'dead'
        0x12d: 'button-circle',         # 301   | 'A'
        0x12e: 'button-cross',          # 302   | 'B'
        0x12f: 'button-square',         # 303   | 'C'
        0x2c0: 'button-ps-logo',        # 704   | (null)
        0x2c1: 'button-?-1',            # 705   | (null)
        0x2c2: 'button-?-2'             # 706   | (null)
        }


_BUTTON_NAME_LOOKUP = {
            0x120 : 'trigger',
            0x121 : 'thumb',
            0x122 : 'thumb2',
            0x123 : 'top',
            0x124 : 'top2',
            0x125 : 'pinkie',
            0x126 : 'base',
            0x127 : 'base2',
            0x128 : 'base3',
            0x129 : 'base4',
            0x12a : 'base5',
            0x12b : 'base6',

            #PS3 sixaxis specific
            0x12c : "triangle",
            0x12d : "circle",
            0x12e : "cross",
            0x12f : 'square',

            0x130 : 'a',
            0x131 : 'b',
            0x132 : 'c',
            0x133 : 'x',
            0x134 : 'y',
            0x135 : 'z',
            0x136 : 'tl',
            0x137 : 'tr',
            0x138 : 'tl2',
            0x139 : 'tr2',
            0x13a : 'select',
            0x13b : 'start',
            0x13c : 'mode',
            0x13d : 'thumbl',
            0x13e : 'thumbr',

            0x220 : 'dpad_up',
            0x221 : 'dpad_down',
            0x222 : 'dpad_left',
            0x223 : 'dpad_right',

            # XBox 360 controller uses these codes.
            0x2c0 : 'dpad_left',
            0x2c1 : 'dpad_right',
            0x2c2 : 'dpad_up',
            0x2c3 : 'dpad_down',
        }


class JoystickDevice:
    ''' Hardware interface to joystick

        Notes:
        * See: https://www.kernel.org/doc/html/v4.16/input/joydev/joystick-api.html

        * This object opens the device in blocking mode.  This means that the read 
          operation will block the thread on which it is running until it registers an
          event.  As a result, this device must definitely be threaded.
          See:
          https://www.kernel.org/doc/html/v4.16/input/joydev/joystick-api.html#reading
          One could open the device in non-blocking mode, but this comes with its own
          headaches (as a queue of events needs then to be processed) and probably does
          not make sense for this use case.

        * The _axes and _buttons variables are lists of names rather than dicts. The
          reason is that the js_event reserves a single byte with which to return the 
          key code.  This is not enough to return some of the hex codes so in the end
          they are mapped internally *in order* to the first bits of the byte.  Hence
          the list.  
    '''
    def __init__(self, device_path='/dev/input/js0'):
        ''' Configure joystick device 

            Arguments

            device_path: str
                path to the physical device
            '''
        self._device_path = device_path

        self.device_name = ''
        self._axes = []
        self._buttons = []

        with open(self._device_path, 'rb') as joystick:
            self._configure(joystick)

    def open(self):
        logging.debug("Opening {}".format(self._device_path))
        self.joystick = open(self._device_path, 'rb')


    def close(self):
        logging.debug("self.joystick.closed = {}".format(self.joystick.closed))
        
        logging.debug("Closing {}".format(self.joystick))
        
        self.joystick.close()


    def poll(self):
        ''' Reads joystick device for signals

            The read returns a wrapped js_event

            The underlying C struct for the js_event contains:
                4-byte timestamp    --> in milliseconds
                2-byte event value  --> {0,1} for buttons, [-MAX_AXIS_VALUE,MAX_AXIS_VALUE] for axes
                1-byte event type   --> JS_EVENT_BUTTON/AXIS/INIT
                1-byte event key    --> axis/button key
            and so expects to be populated with an 8-byte read from the device.


        ''' 
        # in principle a signed short ranges from -32768 to 32767 but the documentation
        # https://www.kernel.org/doc/html/v4.16/input/joydev/joystick-api.html
        # states that only values [-32767,32767] are emitted
        MAX_AXIS_VALUE = 32767

        tag = None
        value = None

        event = self.joystick.read(8)

        if event:
            if TESTING:
                #logging.debug("event hex: {}".format(event.hex()))
                logging.debug("event: {}".format(BitArray(bytes=event)))
                logging.debug("event: {}".format(BitArray(bytes=event).bin))

            # IhBB --> I: unsigned int (4-bytes), h: signed short (2-bytes), B: unsigned char (1-byte)
            timestamp, event_value, event_type, event_key = struct.unpack('IhBB', event)
            if TESTING:
                print("{} - {} - {} - {}".format(timestamp, event_value, event_type, event_key))
            
            if event_type & self._JS_EVENT_INIT:
                pass

            elif event_type & self._JS_EVENT_AXIS:
                tag = self._axes[event_key]
                value = event_value / MAX_AXIS_VALUE

            elif event_type & self._JS_EVENT_BUTTON:
                tag = self._buttons[event_key]
                value = event_value

            else:
                pass

        return tag, value


    def _configure(self, joystick):
        self._JS_EVENT_BUTTON = jsio.retrieve_JS_EVENT_BUTTON()
        self._JS_EVENT_AXIS = jsio.retrieve_JS_EVENT_AXIS()
        self._JS_EVENT_INIT = jsio.retrieve_JS_EVENT_INIT()


        self._retrieve_device_name(joystick)
        self._retrieve_axes(joystick)
        self._retrieve_buttons(joystick)


    def _retrieve_device_name(self, joystick):

        MAX_NAME_LENGTH = 128
        JSIOCGNAME = jsio.retrieve_JSIOCGNAME(MAX_NAME_LENGTH)

        container = array.array('B', [0] * MAX_NAME_LENGTH)

        fcntl.ioctl(joystick, JSIOCGNAME, container)
        
        
        self.device_name = container.tobytes().decode('utf-8')
        logging.debug("device_name: ".format(self.device_name))
        

    def _retrieve_axes(self, joystick):

        JSIOCGAXES = jsio.retrieve_JSIOCGAXES()
        container = array.array('B', [0])
        fcntl.ioctl(joystick, JSIOCGAXES, container)
        nr_axes = container[0]

        JSIOCGAXMAP = jsio.retrieve_JSIOCGAXMAP()
        MAX_NR_AXES = jsio.retrieve_MAX_NR_AXES()
        container = array.array('B', [0] * MAX_NR_AXES)
        fcntl.ioctl(joystick, JSIOCGAXMAP, container)  

        for axis in container[:nr_axes]:
            axis_name = _AXIS_NAME_LOOKUP.get(axis, 'unexpected-({})'.format(axis))
            self._axes.append(axis_name)
            logging.debug("axis_name: ".format(axis_name))


    def _retrieve_buttons(self, joystick):

        JSIOCGBUTTONS = jsio.retrieve_JSIOCGBUTTONS()
        container = array.array('B', [0])
        fcntl.ioctl(joystick, JSIOCGBUTTONS, container) # 
        nr_buttons = container[0]

        JSIOCGBTNMAP = jsio.retrieve_JSIOCGBTNMAP()
        MAX_NR_BUTTONS = jsio.retrieve_MAX_NR_BUTTONS() # 512 in decimal
        container = array.array('H', [0] * MAX_NR_BUTTONS)
        fcntl.ioctl(joystick, JSIOCGBTNMAP, container) # 

        for button in container[:nr_buttons]:
            button_name = _BUTTON_NAME_LOOKUP.get(button, 'unexpected-({})'.format(button))
            self._buttons.append(button_name)
            logging.debug("button_name: ".format(button_name))

    @property
    def _class_string(self):
        return "{} at {}".format(self.__class__.__name__, self.device_name)


# TODO: MJ - {'steering':'human','throttle':'human','recording':False}  
# TODO: MJ - assert d['steering'] in [human','ai']
class Mode:
    ''' A plain old data class that holds the various driving modes

        Members

        steering: str
            either 'human' or 'ai'

        throttle: str
            either 'human' or 'ai'

        recording: bool
            either True or False

    '''
    def __init__(self, steering='human', throttle='human', recording=False):
        self.steering = steering
        self.throttle = throttle
        self.recording = recording



# TODO: set up auto-recording
# TODO: delay (reduced user mobility)
# TODO: publish modes to state as a list/dict with one turned on?? 
#       I don't see the reason right now, but it was a thought
# TODO: Controllers should only produce the raw signal, there should be a
#       separate part to do the manipulation, flipping, scaling, adapting to actuators, etc.

class PS3Controller(BasePart):
    input_keys = ()
    output_keys = ('steering_signal', 'throttle_signal', 'mode')

    def __init__(self, device_path='/dev/input/js0', 
                       steering='human',
                       throttle='human',
                       recording=False,
                       flip_steering=False, 
                       flip_throttle=False,
                       scale_throttle_forward=1.0,
                       scale_throttle_back=1.0,
                       scale_steer_left=1.0,
                       scale_steer_right=1.0,
                       ):

        self.joystick = JoystickDevice(device_path)
        self.mode = {'steering': 'human', 'throttle': 'human', 'recording': False}
        
        self.steering_signal = 0.0
        self.throttle_signal = 0.0
        
        # These values are adjustable
        
        self.adjustment_mode_dict = collections.OrderedDict()
        
        # Construct the dict modes
        self.adjustment_mode_dict['scale_throttle_forward'] = dict()
        self.adjustment_mode_dict['scale_throttle_back'] = dict()
        self.adjustment_mode_dict['scale_steer_left'] = dict()
        self.adjustment_mode_dict['scale_steer_right'] = dict()
        
        # Apply the adjustment values
        self.adjustment_mode_dict['scale_throttle_forward']['value'] = scale_throttle_forward
        self.adjustment_mode_dict['scale_throttle_back']['value'] = scale_throttle_back
        self.adjustment_mode_dict['scale_steer_left']['value'] = scale_steer_left
        self.adjustment_mode_dict['scale_steer_right']['value'] = scale_steer_right

        # Apply the adjustment shifts
        self.adjustment_mode_dict['scale_throttle_forward']['shift'] = 0.01
        self.adjustment_mode_dict['scale_throttle_back']['shift'] = 0.01
        self.adjustment_mode_dict['scale_steer_left']['shift'] = 0.01
        self.adjustment_mode_dict['scale_steer_right']['shift'] = 0.01
                
        pprint.pprint(self.adjustment_mode_dict)
        # This is the adjustment mode
        #self.adjustment_mode_list = ['throttle forward','throttle_back','steer left','steer right']
        
        self.num_adjustment_modes = len(self.adjustment_mode_dict)
        
        self.adjustment_mode_index = 0
        #list(ordered_dict.keys())
        modes = list(self.adjustment_mode_dict.keys())
        self.adjustment_mode = modes[self.adjustment_mode_index]
        
        self.steering_flip = -1.0 if flip_steering else 1.0
        self.throttle_flip = -1.0 if flip_throttle else 1.0

        self.thread = ThreadComponent(self._update)

    def start(self):
        self.joystick.open()
        self.thread.start()


    def transform(self, state):
        state['steering_signal'] = self.steering_signal
        state['throttle_signal'] = self.throttle_signal
        state['mode'] = self.mode
        #logging.debug("TEST".format())
        

    def stop(self):
        logging.debug("Stopping thread".format())
        
        self.thread.stop()
        logging.debug("Closing joystick {}".format(self.joystick))
        
        self.joystick.close()
        #logging.debug("SKIP: Closing joystick!!! This hangs, SKIP!".format())

    def _update(self):
        ''' Update steering and throttle signals

        * axis-thumb-left-x     | steering
        * axis-thumb-right-y    | throttle
        * button-dpad-up        | adjustment 
        * button-dpad-down      | adjustment
        * button-dpad-left      | select adjustment 
        * button-dpad-right     | select adjustment

        * button-triangle       | toggle mode
        * button-circle         | toggle recording
        '''
        
        #THROTTLE_SCALE_SHIFT = 0.01
        #STEERING_SCALE_SHIFT = 0.01
        ADJUSTMENT_SHIFT = 0.01
        #logging.debug("_update".format())

        tag, value = self.joystick.poll()
        
        if TESTING:
            logging.debug("tag {}, value {}".format(tag, value))
        
        if tag == 'axis-thumb-left-x':
            # actuators expect:
            # +ve signal indicates left
            # -ve signal indicated right
            # however, from the joystick interface:
            # +ve value indicates thumb was moved right
            # -ve value indicates thumb was moved left
            # hence the presence of (-value)
            # to allow for an on the fly fudge factor, steering_flip can be activated
            #print()
            #self.steering_signal = self.steering_scale * self.steering_flip * (-value)
            self.steering_signal = self.steering_flip * (-value)
            #print("Steer:",self.steering_signal)

        elif tag == 'axis-thumb-right-y':
            # actuators expect:
            # +ve signal indicates forward
            # -ve signal indicates reverse
            # however, from the joystick interface:
            # +ve value indicate the thumb was pulled back
            # -ve value indicates thumb was pushed forward
            # hence the presence of (-value)
            # to allow for an on the fly fudge factor, throttle_flip can be activated
            #self.throttle_signal =  self.throttle_scale * self.throttle_flip * (-value)
            
            # +ve throttle is +value (more intuitive!)
            value = -value
             
            #print("Throttle original value",value)
            if value > 0:
                multiplier = self.adjustment_mode_dict['scale_throttle_forward']['value']
                self.throttle_signal = multiplier * self.throttle_flip * (value)
                #print("Throttle forward value",self.throttle_signal)
            elif value < 0:
                multiplier = self.adjustment_mode_dict['scale_throttle_back']['value']
                self.throttle_signal = multiplier * self.throttle_flip * (value)
                print("Throttle reverse value",self.throttle_signal)
            else: 
                self.throttle_signal =  self.throttle_flip * (value)
                #print("Throttle neutral value",self.throttle_signal)
        
        #--- dpad-up
        elif tag == 'button-dpad-up' and value == 1:
            #self.throttle_scale = min(1.0, self.throttle_scale + THROTTLE_SCALE_SHIFT)
            #print("")
            #logging.debug("{} throttle_scale = {:.2f}".format(tag,self.throttle_scale))
            
            adjust = self.adjustment_mode_dict[self.adjustment_mode]['shift']
            old_value = self.adjustment_mode_dict[self.adjustment_mode]['value']
            new_value = old_value + adjust
            self.adjustment_mode_dict[self.adjustment_mode]['value'] = new_value
            logging.debug("Adjusted {} to {:.2f}".format(self.adjustment_mode, new_value))

        #--- dpad-down
        elif tag == 'button-dpad-down' and value == 1:
            #self.throttle_scale = max(0.0, self.throttle_scale - THROTTLE_SCALE_SHIFT)
            #logging.debug("{} throttle_scale = {:.2f}".format(tag,self.throttle_scale))
            adjust = -self.adjustment_mode_dict[self.adjustment_mode]['shift']
            old_value = self.adjustment_mode_dict[self.adjustment_mode]['value']
            new_value = old_value + adjust
            self.adjustment_mode_dict[self.adjustment_mode]['value'] = new_value
            logging.debug("Adjusted {} to {:.2f}".format(self.adjustment_mode, new_value))
            
        #--- dpad-right
        elif tag == 'button-dpad-right' and value == 1:
            self.adjustment_mode_index += 1
            self.adjustment_mode_index = self.adjustment_mode_index % self.num_adjustment_modes
            modes = list(self.adjustment_mode_dict.keys())
            #print(modes,self.adjustment_mode_index)
            self.adjustment_mode = modes[self.adjustment_mode_index]
            #self.adjustment_mode = self.adjustment_mode_dict[self.adjustment_mode_index]
            #print(self.adjustment_mode_index, self.adjustment_mode)
            logging.debug("Adjustment mode {}, {}".format(self.adjustment_mode_index,self.adjustment_mode))            

        #--- dpad-left
        elif tag == 'button-dpad-left' and value == 1:
            self.adjustment_mode_index += -1
            self.adjustment_mode_index = self.adjustment_mode_index % self.num_adjustment_modes
            modes = list(self.adjustment_mode_dict.keys())
            #print(modes,self.adjustment_mode_index)
            self.adjustment_mode = modes[self.adjustment_mode_index]
            #self.adjustment_mode = self.adjustment_mode_dict[self.adjustment_mode_index]
            logging.debug("Adjustment mode {}, {}".format(self.adjustment_mode_index,self.adjustment_mode))
            
        elif tag == 'button-triangle' and value == 1:
            self.mode['steering'] = 'human'
            self.mode['throttle'] = 'human'
            logging.debug("{} mode = {}".format(tag,self.mode))
            
        elif tag == 'button-square' and value == 1:
            self.mode['steering'] = 'human'
            self.mode['throttle'] = 'ai'
            logging.debug("{} mode  = {}".format(tag,self.mode))
            
        elif tag == 'button-circle' and value == 1:
            self.mode['steering'] = 'ai'
            self.mode['throttle'] = 'human'
            logging.debug("{} mode = {}".format(tag,self.mode))
            
        elif tag == 'button-cross' and value == 1:
            self.mode['steering'] = 'ai'
            self.mode['throttle'] = 'ai'
            logging.debug("{} mode = {}".format(tag,self.mode))
            
        elif tag == 'button-select' and value == 1:
            self.mode['recording'] = not self.mode['recording']
            logging.debug("{} mode = {}".format(tag,self.mode))
            
