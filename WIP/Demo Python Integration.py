
import pandas as pa
import pip
import numpy as np
#pip.main(['install', 'aptecoPythonUtilities'])

import aptecoPythonUtilities as apu
import sys

def inputFile() :  return sys.argv[1]



def read():
    print("Starting to read input data...")

    try:
        print("Trying to read data as comma separated...")
        hRaw = pa.read_csv(inputFile())
    except:
        print ("Unable to read as comma separated.")
        try:
            print ("Trying to read as tab separated.")
            hRaw = pa.read_csv(inputFile(), sep='\t')
        except:
            print ("Unable to read as tab separated.")

    print("... read file with {} records.".format(len(hRaw)))

    return hRaw



def summary(df):
    apu.utils_explore.overview(df)



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