## Handle Event
## Read Event from event17

import asyncio
import evdev
from evdev import InputDevice, categorize, ecodes
import datetime

last_time = datetime.datetime.now()

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# print(len(devices))
dev = InputDevice('/dev/input/event' + str(len(devices)-1))

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


## using ascync
# async def helper(dev):
#     async for ev in dev.async_read_loop():
#         # print(repr(ev))
#         print('event')

# loop = asyncio.get_event_loop()
# loop.run_until_complete(helper(dev))
