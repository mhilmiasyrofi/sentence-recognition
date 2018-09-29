import bluetooth

print("Start discovering....")

nearby_devices = None

try:
    nearby_devices = bluetooth.discover_devices()

    print("Found devices:")

    print(len(nearby_devices))

    # for bdaddr in nearby_devices:
    #     print(bluetooth.lookup_name(bdaddr), "with address:", bdaddr)
except OSError as err:
    print("OS error: {0}".format(err))