import matplotlib.pyplot as plt
import pandas as pa
import numpy as np
import seaborn as sns


# class MyExp :
def unique(df):
    for c in df.columns.values:
        print(c, df[c].nunique())


def overview(df):
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    unique(df)


def detail(df):
    for c in df.columns.values:
        print(c)
        print(df[c].value_counts())
        print()
    print()
    print(df.head())
    print()


# df=train
# overview(df)
# unique(df)


def hist(df, byVar=None, bins=5):
    df.hist(column=byVar, bins=bins)
    plt.show()


# hist(df)
# hist(df,"Year")
# hist(df,"IsHigh")
# hist(df,["Year", "Cost"])


def heat2D(df, rowVar, colVar, ofVar, agg=np.mean):
    piv = pa.pivot_table(df, index=[rowVar], columns=[colVar], values=ofVar, aggfunc=agg, fill_value=0)
    sns.heatmap(piv)
    plt.show()


# heat2D(df, "Year", "Destination", "Cost", np.mean)


def freq2D(df, rowVar, colVar):
    df2 = df
    df2["_dummy_"] = 1
    piv = pa.pivot_table(df2, index=[rowVar], columns=colVar, values="_dummy_", aggfunc=np.sum, fill_value=0)
    sns.heatmap(piv)
    plt.show()

# freq2D(df, "Year", "Destination")






