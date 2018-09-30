## Handle Event
## Read Event from event17

import asyncio
import evdev
from evdev import InputDevice, categorize, ecodes
import datetime

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# print(len(devices))
dev = InputDevice('/dev/input/event' + str(len(devices)-1))

import numpy as np
import cv2

last_time = datetime.datetime.now()
cap = cv2.VideoCapture(0)
i = 0

delta = datetime.datetime.now() - last_time
timer = delta.seconds + delta.microseconds/1E6
# print(combined)
while timer < 3 :
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = gray
    cv2.imshow('frame', output) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    delta = datetime.datetime.now() - last_time
    timer = delta.seconds + delta.microseconds/1E6


last_time = datetime.datetime.now()

## using sync approach
for event in dev.read_loop():
    if event.type == ecodes.EV_KEY:
        # print(categorize(event))
        delta = datetime.datetime.now() - last_time
        combined = delta.seconds + delta.microseconds/1E6
        # print(combined)
        if combined > 2 :
            last_time = datetime.datetime.now()
            print("event")
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output = gray
            cv2.imshow('frame', output) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Display the resulting frame
    cv2.imshow('frame', output) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# last_time = datetime.datetime.now()

# # using ascync
# async def helper(dev):
#     async for ev in dev.async_read_loop():
#         # print(repr(ev))
#         delta = datetime.datetime.now() - last_time
#         combined = delta.seconds + delta.microseconds/1E6
#         # print(combined)
#         if combined > 2 :
#             last_time = datetime.datetime.now()
#             print('event')

# loop = asyncio.get_event_loop()
# loop.run_until_complete(helper(dev))