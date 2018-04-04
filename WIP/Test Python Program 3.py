
import pandas as pa
import pip
pip.main(['install', 'aptecoPythonUtilities'])

import utils_explore

def read():
    print("Starting to read bookings...")

    hRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Holidays\Bookings for All People.csv")

    print("... read file with {} bookings.".format(len(hRaw)))

    return hRaw



def summary(df):
    utils_explore.overview(df)



raw = read()
summary(raw)





# Define Class and use __main__ to run code
class MyProg():

    def __init__(self, x):
        self.x=x
        print ("Using {}".format(x))

    def run(self):
        hRaw = read()
        print(self.x * self.x)

if __name__ == '__main__':
    prog = MyProg(10)

    prog.run()