## Handle Event
## Read Event from event17

import asyncio
import evdev
from evdev import InputDevice, categorize, ecodes
import datetime

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
dev = InputDevice('/dev/input/event' + str(len(devices)-4))

import numpy as np
import cv2

import _thread

state = False

def readInput():
    global state
    last_time = datetime.datetime.now()
    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            # print(categorize(event))
            delta = datetime.datetime.now() - last_time
            combined = delta.seconds + delta.microseconds/1E6
            if combined > 2 :
                last_time = datetime.datetime.now()
                # print("event")
                state = True
    print("finish")

_thread.start_new_thread( readInput, () )

cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # output = gray
    output = frame
    cv2.imshow('frame', output) 
    if state == True :
        print("event")
        state = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
