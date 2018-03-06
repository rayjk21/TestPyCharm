
import pandas as pa
import numpy as np


def groupArrays(df, byVar, ofVar):
        cols = df.filter([byVar,ofVar])
        keys,values=cols.sort_values(byVar).values.T
        ukeys,index=np.unique(keys,True)
        arrays=np.split(values,index[1:])
        df2=pa.DataFrame({byVar:ukeys,ofVar:[list(a) for a in arrays]})
        return df2

