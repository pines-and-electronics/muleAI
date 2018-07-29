#!/usr/bin/python2
import struct
with open("/dev/input/js0", "rb") as f:
    while True:
        a = f.read(8)
        t, value, code, index = struct.unpack("<ihbb", a) # 4 bytes, 2 bytes, 1 byte, 1 byte
        # t: time in ms
        # index: button/axis number (0 for x-axis)
        # code: 1 for buttons, 2 for axis
        # value: axis position, 0 for center, 1 for buttonpress, 0 for button release
        print("t: {:10d} ms, value: {:6d}, code: {:1d}, index: {:1d}".format(t, value, code, index))