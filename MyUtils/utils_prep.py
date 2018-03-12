
import pandas as pa
import numpy as np


def groupArrays(df, byVar, ofVar):
    cols = df[[byVar,ofVar]]
    keys,values=cols.sort_values(byVar).values.T
    ukeys,index=np.unique(keys,True)
    arrays=np.split(values,index[1:])
    df2=pa.DataFrame({byVar:ukeys,ofVar:[list(a) for a in arrays]})
    return df2


def xPerY(df, ofVar, byVar, renameTo=None, sort=True):
    per = df[[ofVar, byVar]].drop_duplicates().groupby(byVar).agg({ofVar:['count']})
    if renameTo==None: renameTo = "{} Per {}".format(ofVar, byVar)
    per.columns=[renameTo]
    if sort: per = per.sort_values(by=renameTo, ascending=False)
    return per[renameTo]