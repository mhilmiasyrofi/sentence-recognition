## Handle Event
## Read Event from event17

import asyncio
from evdev import InputDevice, categorize, ecodes

dev = InputDevice('/dev/input/event17')

async def helper(dev):
    async for ev in dev.async_read_loop():
        print(repr(ev))

loop = asyncio.get_event_loop()
loop.run_until_complete(helper(dev))
