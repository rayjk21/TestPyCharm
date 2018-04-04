
import pandas as pa
import pip



def read():
    print("Starting to read bookings...")

    hRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Holidays\Bookings for All People.csv")

    print("... read file with {} bookings.".format(len(hRaw)))

    return hRaw



read()



