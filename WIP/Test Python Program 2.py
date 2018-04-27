
import pandas as pa




def read():
    print("Starting to read bookings...")

    hRaw = pa.read_csv(r"\\aptwarnas1\shareddata\develop\Data\Holidays\Bookings for All People.csv")

    print("... read file with {} bookings.".format(len(hRaw)))

    return hRaw



read()



