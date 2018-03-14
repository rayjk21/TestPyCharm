
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import MyUtils.Embeddings as myEmb
import MyUtils.utils_explore as myExp
import MyUtils.utils_prep as myPrep
import MyUtils.utils_plot as myPlot
import MyUtils.utils_nn as myNn
import math
import itertools

import sklearn.preprocessing as skPrep

pa.set_option('max_rows', 11)
pa.set_option('expand_frame_repr', False)

###################################################################################
# Retail
###################################################################################

rRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Retail\All Cust Items Data Grid.txt", sep='\t')
rRaw["Department"].astype(str)
rRaw['Product Group'].astype(str)
rRaw['Product Group'] = "_" + rRaw['Product Group'].astype(str)

# Encode long CUSTIDs as consecutive integers
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(rRaw["CUSTID"])
rRaw["CustIx"] = np.vectorize(lambda x: 'c'+str(x))(le.transform(rRaw["CUSTID"]))



# Drop very small/large customers and baskets
productsPerBasket   = myPrep.xPerY(rRaw,"Product Group", "BASKET ID","PperB" ).where(lambda x: (x>1) & (x<50)).dropna()
basketsPerCustomer  = myPrep.xPerY(rRaw,"BASKET ID","CUSTID", "BperC" ).where(lambda x: (x>2) & (x<20)).dropna()
#productsPerCustomer = myPrep.xPerY(rRaw,"Product Group","CUSTID", "PperC" ).where(lambda x: (x>1) & (x<10)).dropna().head(100)
allCusts = basketsPerCustomer.index.values

#productsPerBasket.plot.hist(range(0,50,5))
#basketsPerCustomer.plot.hist(range(0,20,2))

df1 = pa.merge(rRaw, basketsPerCustomer.to_frame(), how='inner', left_on=["CUSTID"], right_index=True)
df2 = pa.merge(df1, productsPerBasket.to_frame(),  how='inner', left_on=["BASKET ID"], right_index=True)

#df2 = pa.merge(rRaw, productsPerCustomer.to_frame(),  how='inner', left_on=["CUSTID"], right_index=True)
keepCols = ['CUSTID', 'BASKET ID', 'Trans Datetime', 'PRICE PAID',
            'Product Group', 'Super Group', 'Department',
            'BperC', 'PperB','Lifestage']

dfSubSet = df2.set_index("CustIx")[keepCols].dropna()

myExp.overview(dfSubSet)
myExp.detail(dfSubSet)

myExp.overview(rRaw)
myExp.detail(rRaw)

dfSubSet["PperB"].plot.hist()






###################### Create feature vectors
df = dfSubSet.reset_index()
deptEmb = myEmb.CreateFromDf (df, 'CustIx', 'Department')
#deptEmb.plotAll()
pgpEmb = myEmb.CreateFromDf (df, 'CustIx', 'Product Group')

# Merge on to Subset
df3 = pa.merge(df, pgpEmb.df, on='Product Group', how='inner')

dfFeature = df3.set_index('CustIx')






############# Investigate Target Products
df = dfFeature
productPerDept = myPrep.xPerY(rRaw,'Product Group', 'Department' )
productToDept = rRaw[['Product Group', 'Department']].drop_duplicates()
uDepts = list(productPerDept.index.values)
pgpEmb.addLookup ("Department", productToDept)

chinese = pgpEmb.df["Product Group"].str.contains("CHINESE")
indian  = pgpEmb.df["Product Group"].str.contains("INDIAN")
cou     = pgpEmb.df["Product Group"].str.contains("COU ")
target = (chinese | indian) & (~cou)
targetProducts = list(pgpEmb.df.loc[target]["Product Group"])

# Oriental is identified more by F3 & F4
pgpEmb.plotGroup("Department","MEAL SOLUTIONS", ["F1","F2"], highlight=target)
pgpEmb.plotGroup("Department","MEAL SOLUTIONS", ["F3","F4"], highlight=target)
pgpEmb.plotGroups("Department",uDepts, ["F3","F4"], highlight=target)








#################### Mark target products
df["TargetProduct"] = np.where(df["Product Group"].isin(targetProducts), True, False)
targetCustomers    = df[df["TargetProduct"]==True].dropna().index.drop_duplicates()
df["TargetCustomer"] = np.where(df.index.isin(targetCustomers), True, False)

flags = df["TargetCustomer"]
flag0 = list(flags[flags==False].index.drop_duplicates())
flag1 = list(flags[flags==True].index.drop_duplicates())


print("Target Customers:", len(targetCustomers))
print("Target Customers:", len(flag1))
print("Non Target Customers:", len(flag0))


def CheckSummary():
    df["Freq"] = 1
    pa1 = pa.pivot_table(df, index=["TargetCustomer"], columns=["TargetProduct"], values=["CUSTID"], aggfunc=lambda x: len(x.unique()))
    print(pa1)

    cust_sum = pa.pivot_table(df, index=["TargetCustomer", "CUSTID"], columns=["TargetProduct"], aggfunc={"Freq" : 'count'}).reset_index().set_index("CUSTID")
    print(cust_sum)
    cust_sum.columns = cust_sum.columns.droplevel(0)

    cust_sum["PctTarget"] = cust_sum[True] / (cust_sum[True] + cust_sum[False])

    cust_sum.nlargest(columns=["PctTarget"], n=10)
    cust_sum.nsmallest(columns=["PctTarget"], n=10)
    cust_sum.sort_values(["PctTarget"], ascending=False)

def CheckCustomer(ix):
    cust = df.loc[targetCustomers[ix]]
    prods = cust.where(cust["TargetProduct"]==True).dropna()
    print(prods)

df = dfFeature







#################### Preprocess data  2- standardise the inputs as Pct or normalised

def pre_process_data_OLD(df, byVar, dims=["F3", "F4"], nBins=4, type='Pct', ofVar=None, labels=False, offset='Min'):
    """
    Preprocess data for Neural Network
        - Removes the TargetProducts
        nBins = Number of bins for each of the dimensions
        type = 'Norm' to use normalised ratings
        type = 'Pct'  to use the percentage of ratings

        :rtype:
            Pandas dataframe
            Returns 2D array (shape nBins,nBins) for each byVar
    """
    xCol, yCol = dims
    xLabels, yLabels = None, None
    xCuts = pa.cut(df[xCol], bins=nBins, retbins=True)[1]
    yCuts = pa.cut(df[yCol], bins=nBins, retbins=True)[1]

    if (not labels):
        xLabels = range(nBins)
        yLabels = range(nBins)

    df["X"] = pa.cut(df[xCol], bins=xCuts, labels=xLabels).astype(str)
    df["Y"] = pa.cut(df[yCol], bins=yCuts, labels=yLabels).astype(str)
    df["Freq"] = 1

    def calcPct(df, byVar, dims, nBins, labels=False, offset='Min'):
        """
            Divides the dimensions, dims, into nBins each
            Does a freq count of rows in each bin for each level of byVar and overall
            Calculates for each level of byVar, the number of rows for that level
            in each bin as a percentage of all rows

            Inputs:
                df=df
                nBins=3
                dims = ["F1", "F2"]
                byVar="CUSTID"
                xCol, yCol = dims
                labels = False
        """

        eachPiv = pa.pivot_table(df, index=[byVar, "Y"], columns=["X"], values="Freq", aggfunc=np.sum, dropna=False)
        basePiv = pa.pivot_table(df, index=["Y"], columns=["X"], values="Freq", aggfunc=np.sum)
        pct = (eachPiv / basePiv) * 10000

        # Standardise within each customer
        uCust = pct.index.get_level_values(0).drop_duplicates()
        nCust = len(uCust)
        dShape = nCust,nBins*nBins

        pctM = pct.as_matrix().reshape(dShape)
        eachPctCount = pct.groupby(byVar).count().sum(axis=1).as_matrix().reshape(nCust,1)
        eachPctSum   = pct.groupby(byVar).sum().sum(axis=1).as_matrix().reshape(nCust,1)
        eachPctMin   = pct.groupby(byVar).min().min(axis=1).as_matrix().reshape(nCust,1)
        eachPctMax   = pct.groupby(byVar).max().max(axis=1).as_matrix().reshape(nCust,1)
        eachPctRng   = eachPctMax - eachPctMin

        eachPctMean  = eachPctSum / eachPctCount

        # Standardise, setting to 0.5 if there is only one value (hence range==0)
        eachPctRng[eachPctRng==0] = 1.0
        if (offset == 'Mean'):
            reset = (pctM - eachPctMean)
        if (offset == 'Min'):
            reset = (pctM - eachPctMin)
        eachStd = (reset / eachPctRng).reshape(nCust,nBins,nBins)
        eachStd[eachStd==0] = 0.5
        np.nan_to_num(eachStd, copy=False)


        stds = eachStd.reshape(nCust * nBins, nBins)
        custIx = np.repeat(uCust, nBins)
        cols = [byVar] + list(xLabels)
        dfX = pa.DataFrame(np.column_stack([custIx, stds]),columns=cols)
        dfX = dfX.set_index(byVar)
        return dfX

    def calcNorm(df, byVar, ofVar, dims, nBins, labels=False):
        """

            Divides the dimensions, dims, into nBins each
            Sums the ofVar in each bin for each level of byVar and overall
            Calculates for each level of byVar, the sum for that level
            in each bin divided by the sum for all rows

            df=df
            nBins=4
            dims = ["F1", "F2"]
            byVar="CUSTID"
            ofVar="PRICE PAID"
            xCol, yCol = dims
            labels = False
        """
        allMean = df[ofVar].mean()
        allStd = df[ofVar].std()
        cellMeans = pa.pivot_table(df, index=[byVar, "Y"], columns=["X"], values=ofVar, aggfunc=np.mean, dropna=False)
        cellMeans = cellMeans.fillna(allMean)
        cellNorms = (cellMeans - allMean) / allStd

        return cellNorms

    # Can't use target products
    #useDf = df
    useDf = df[~df["TargetProduct"]]

    if type == 'Pct':
        dfX = calcPct(useDf, byVar, dims, nBins, offset=offset)

    if type == 'Norm':
        dfX = calcNorm(useDf, byVar, ofVar, dims, nBins)

    dfY = useDf["TargetCustomer"].to_frame().reset_index().drop_duplicates()
    dfY = dfY.sort_values(byVar).set_index(byVar)

    return dfX, dfY


def pre_process_data(df, byVar, dims=["F3", "F4"], nBins=4, type='Count', ofVar=None, how='Index', omitTarget=True) -> pa.DataFrame:
    """
    Preprocess data for Neural Network
        - Removes the TargetProducts
        nBins = Number of bins for each of the dimensions
        type = 'Value', how = 'Mean', ofVar = 'Rating' to use mean ratings
        type = 'Count', how = 'MinRng' to base on Ix of count, offset by Min and scaled by Range

    dims = ["F1", "F2"]
    nBins = 2
    df = dfFeature
    byVar = 'CUSTID'

        :rtype:
            Pandas dataframe
            Returns 2D array (shape nBins,nBins) for each byVar
    """

    def createCuts(df, dimName, cutName, nBins=4, showBands=False):
        cuts = pa.cut(df[dimName], bins=nBins, retbins=True)[1]

        if (not showBands):
            labels = range(nBins)
        else:
            labels = None

        df[cutName] = pa.cut(df[dimName], bins=cuts, labels=labels).astype(str)

    def norm(eachPiv, basePiv, how):

        # Standardise within each customer
        uCust  = eachPiv.index
        nCust  = len(uCust)
        nCells = nBins**(len(dims))

        eachM = eachPiv.as_matrix().reshape(nCust, nCells)
        baseM = basePiv.as_matrix().reshape(1, nCells)


        baseSum   = np.nansum(baseM, axis=1)
        basePct   = baseM / baseSum

        eachCount = np.count_nonzero(~np.isnan(eachM), axis=1).reshape(nCust,1)
        eachSum   = np.nansum(eachM, axis=1).reshape(nCust,1)
        eachMean  = eachSum / eachCount
        eachPct   = eachM / eachSum            # Pct-total in each cell

        eachMin   = np.nanmin(eachM, axis=1).reshape(nCust,1)
        eachMax   = np.nanmax(eachM, axis=1).reshape(nCust,1)
        eachRng   = eachMax - eachMin
        eachRng[eachRng == 0] = 1.0

        ixNan = np.where(np.isnan(eachM))

        eachIx = (eachPct / basePct)   # Ix of pct in each cell compared to overall
        eachIx[ixNan] = 1.0
        vLog = np.vectorize(lambda x: math.log(x,2))
        eachIx = vLog(eachIx)

        eachMinRng  = (eachM - eachMin)  / eachRng    # 0 = Min for person,  1 = Max for person
        eachMinRng[ixNan] = 0                         # -1 for None

        eachVMax  = eachM   / eachMax                # 1 = Max for person
        eachVMax[ixNan] = 0                          # 0 for None



        eachMeanRng = (eachM - eachMean) / eachRng    # 0 = Mean for person, +/- = over/under mean
        eachMeanRng[ixNan] = 0

        if (how == 'MinRng'):
            norm = eachMinRng
        elif (how == 'MeanRng'):
            norm = eachMeanRng
        elif (how == 'Index'):
            norm = eachIx
        elif (how == 'Max'):
            norm = eachVMax



        dfX = pa.DataFrame(data=norm, columns=eachPiv.columns, index=eachPiv.index)

        return dfX

    def normCount(df, byVar, dims, nBins, labels=False, how='MinRng'):
        """
            Divides the dimensions, dims, into nBins each
            Does a freq count of rows in each bin for each level of byVar and overall
            Calculates for each level of byVar, the number of rows for that level
            in each bin as a percentage of all rows

            Inputs:
                df=df
                nBins=3
                dims = ["F1", "F2"]
                byVar="CUSTID"
                xCol, yCol = dims
                labels = False
        """

        eachPiv = pa.pivot_table(df,  index=[byVar],   columns=cutNames, values="Freq", aggfunc=np.sum, dropna=False)
        basePiv = pa.pivot_table(df,  index=[baseVar], columns=cutNames, values="Freq", aggfunc=np.sum, dropna=False).fillna(1)

        norms = norm(eachPiv, basePiv, how)

        return norms

    def normValue(df, byVar, ofVar, dims, nBins, labels=False, how='Mean'):
        """

            Divides the dimensions, dims, into nBins each
            Sums the ofVar in each bin for each level of byVar and overall
            Calculates for each level of byVar, the sum for that level
            in each bin divided by the sum for all rows

            df=df
            nBins=4
            dims = ["F1", "F2"]
            byVar="CUSTID"
            ofVar="PRICE PAID"
            xCol, yCol = dims
            labels = False
        """

        cellSums  = pa.pivot_table(df, index=[byVar],   columns=cutNames, values=ofVar, aggfunc=np.sum, dropna=False)
        cellMeans = pa.pivot_table(df, index=[byVar],   columns=cutNames, values=ofVar, aggfunc=np.mean, dropna=False)

        allSum    = pa.pivot_table(df, index=[baseVar],   columns=cutNames, values=ofVar, aggfunc=np.sum, dropna=False)
        allMean   = pa.pivot_table(df, index=[baseVar],   columns=cutNames, values=ofVar, aggfunc=np.mean, dropna=False)

        if (how=='Sum'):
            # Makes more sense for retail
            # - compares total spend per cell for a person, to that overall in the cell
            norms = norm(cellSums, allSum, 'MinRng')
        if (how=='Mean'):
            # Makes more sense for movies
            # - compares avg rating per cell for a person, to that overall in the cell
            norms = norm(cellMeans, allMean, 'MinRng')

        return norms

    # Band the dimensions
    cutNames = []
    for d in dims:
        cutName = d + "_"
        createCuts(df, d, cutName, nBins=nBins)
        cutNames.append(cutName)



    df["Freq"] = 1
    baseVar = "_Base_"
    df[baseVar] = 'Base'

    # Can't use target products
    if omitTarget:
        useDf = df[~df["TargetProduct"]]
    else:
        useDf = df

    if type == 'Count':
        dfX = normCount(useDf, byVar, dims, nBins, how=how)
    elif type == 'Value':
        dfX = normValue(useDf, byVar, ofVar, dims, nBins, how=how)
    else:
        print("Unrecognised type '{}'".format(type))
        return

    dfY = useDf["TargetCustomer"].to_frame().reset_index().drop_duplicates()
    dfY = dfY.sort_values(byVar).set_index(byVar)

    return dfX, dfY

dfX, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='Index')
dfX, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='MinRng')

myExp.overview(dfX)
myExp.detail(dfX)
myExp.overview(dfY)




###################### Plot individual member who is/isnt in the target

def plot1(mId, dfX, df, lbl, dims=["F3", "F4"], colorVar=None, highlight=[], show=True, hideAxes=False, r=1, c=2, p1=1, p2=2):
    """
        mId = 'c1'
        df  = Data for each member for each Title
        dfX = X values for modelling, summarised for each member over titles
        color: Specify item level variable for use as thematic (or None for default)
    """

    def highlightPoints():
        tData = mData[mData[lbl].isin(highlight)]
        myPlot.scatterDf(tData, dims, color='Red', size=20, hideAxes=True)

    # Individual points
    plt.subplot(r,c,p1)
    mData = df.loc[mId]
    title = ("Id: " + str(mId))

    xDim,yDim = dims
    xMin = np.min(df[xDim])
    xMax = np.max(df[xDim])
    yMin = np.min(df[yDim])
    yMax = np.max(df[yDim])

    if colorVar != None : c = mData[colorVar].values
    else: color='Blue'
    #myPlot.scatterDf(mData, dims, lbl, color=color, size=20, fSize=6, hideAxes=True)
    myPlot.scatterDf(mData, dims, lbl, color=color, size=20, fSize=6, hideAxes=hideAxes, xRange=(xMin,xMax), yRange=(yMin,yMax))
    highlightPoints()

    # Heatmap of model input
    plt.subplot(r,c,p2)
    hData = myPlot.make2D(dfX.loc[mId].as_matrix()).T
    myPlot.heatMap(hData, show_legend=True, show=False, diverge=True, hideAxes=hideAxes)
    #myPlot.heatMap(hData, xCol=dims[0], yCol=dims[1], show_legend=True, show=False, diverge=True, hideAxes=hideAxes)

    if show: plt.show()

def plotN(custs, highlight=[]):
    n=len(custs)
    for i,cust in enumerate(custs):
        print("{} {}".format(i,cust))
        plot1(cust, dfX, df, "Product Group", show=False, r=2,c=n,p1=i+1,p2=i+1+n, highlight=highlight, hideAxes=True)
    plt.show()

# Display comparison heat plots
def plotCompare(dfX, flag0, flag1, nGrid=3, showTitle=False):
    nn = nGrid
    p = 0
    for i in range(nn):
        for j in range(nn):
            plt.subplot2grid((nn, nn * 2), (i, j))
            hData0 = myPlot.make2D(dfX.loc[flag0[p]].as_matrix()).T
            hData1 = myPlot.make2D(dfX.loc[flag1[p]].as_matrix()).T
            print("Row {}, Column {}, Flag {}, Id {}".format(i,j,0,flag0[p]))
            myPlot.heatMap(hData0, show_legend=False, show_axisName=False, show_axisLbls=False, show=False, diverge=True)
            plt.subplot2grid((nn, nn * 2), (i, nn+j))
            print("Row {}, Column {}, Flag {}, Id {}".format(i, j, 1, flag1[p]))
            myPlot.heatMap(hData1, show_legend=False, show_axisName=False, show_axisLbls=False, show=False, diverge=True)
            p = p + 1
    plt.show()


dfXi, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='Index')
dfXmr, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='MinRng')
dfXm, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='Max')


myPlot.compareDistributions(dfXi, dfY, ignore=0)
myPlot.compareDistributions(dfXmr, dfY, ignore=0)
myPlot.compareDistributions(dfXm, dfY, ignore=0)


dfXi.loc['c1'].as_matrix()
dfXmr.loc['c1'].as_matrix()
dfXm.loc['c1'].as_matrix()

plot1('c19804', dfXi, df, "Product Group", highlight=targetProducts)
plot1('c1', dfXm, df, "Product Group", highlight=targetProducts)



plot1(flag0[0], dfX, df, "Product Group", highlight=targetProducts, dims=["F1", "F2"])
plot1(flag0[15], dfX, df, "Product Group", highlight=targetProducts, dims=["F3", "F4"])

plot1(flag1[0], dfX, df, "Product Group", highlight=targetProducts, dims=["F1", "F2"])
plot1(flag1[0], dfX, df, "Product Group", highlight=targetProducts, dims=["F3", "F4"])

plotN(flag1[10:15], highlight=targetProducts)

plotCompare(dfXi, flag0, flag1, nGrid=5, showTitle=True)
plotCompare(dfXmr, flag0, flag1, nGrid=5)









#################  NN

import keras
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
import sklearn

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow as tf
import seaborn as sns

from keras.callbacks import History


def ae1(nBins=10):
    input = Input(shape=(nBins, nBins, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(8,  (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1,  (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(49, activation='relu')(x)
    x = Dense(25, activation='relu')(x)
    # "decoded" is the lossy reconstruction of the input

    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = x
    #encoding_dim = encoded.get_shape().as_list()
    x = Dense(49, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Reshape((10,10,1))(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    #x = UpSampling2D((2, 2))(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2D(8 , (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = x
    encoder     = Model(input, encoded)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss=keras.losses.mse)
    return encoder, autoencoder

def ae2(nBins=10, xBins=None, yBins=None, encode_size=25):
    xBins, yBins = myNn.setBins(nBins, xBins, yBins)
    input = Input(shape=(yBins, xBins, 1))
    x = Conv2D(8,  (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(1,  (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(encode_size*2, activation='relu')(x)
    x = Dense(encode_size, activation='relu', name='Encoding_Layer')(x)
    encoded = x

    x = Dense(encode_size*2, activation='relu')(x)
    x = Dense(yBins*xBins, activation='relu')(x)
    x = Reshape((yBins,xBins,1))(x)
    x = Conv2D(8 , (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = x

    encoder     = Model(input, encoded)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss=keras.losses.mse)

    return encoder, autoencoder

def ae3(nBins=10, xBins=None, yBins=None, encode_size=25):
    xBins, yBins = myNn.setBins(nBins, xBins, yBins)
    input = Input(shape=(yBins, xBins, 1))
    x = Conv2D(8,  (3, 3), activation='relu', padding='same')(input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    shape = int(yBins/2), int(xBins/2), 1
    x = Flatten()(x)

    x = Dense(encode_size*2, activation='relu')(x)
    encoded = Dense(encode_size, activation='relu', activity_regularizer=keras.regularizers.l1(10e-7), name='Encoding_Layer')(x)
    x = Dense(encode_size*2, activation='relu')(encoded)
    x = Dense(shape[0] * shape[1], activation='relu')(x)
    x = Reshape(shape)(x)

    x = Conv2D(8 , (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    decoded = x

    encoder     = Model(input, encoded)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss=keras.losses.mse)

    return encoder, autoencoder

def ae4(input_size, encode_size, layer_sizes=[256, 128]):
    input_shape = (1,1,input_size)
    input = Input(shape=(input_shape))

    x = Dense(layer_sizes[0], activation='relu')(input)
    x = Dense(layer_sizes[1], activation='relu')(x)
    x = Dense(encode_size, activation='relu', activity_regularizer=keras.regularizers.l1(10e-7), name='Encoding_Layer')(x)

    encoded = x
    #encoding_dim = encoded.get_shape().as_list()
    x = Dense(layer_sizes[1], activation='relu')(x)
    x = Dense(layer_sizes[0], activation='relu')(x)
    x = Dense(input_size, activation='sigmoid')(x)
    decoded = x

    encoder     = Model(input, encoded)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss=keras.losses.mse)

    return encoder, autoencoder

def ae4b(input_size, encode_size, layer_sizes=[256, 128], pen = None, loss=keras.losses.mse):
    input_shape = (input_size,)
    input = Input(shape=(input_shape))

    if pen is not None:
        reg = keras.regularizers.l1(pen)
    else:
        reg = None

    x = Dense(layer_sizes[0], activation='relu')(input)
    x = Dense(layer_sizes[1], activation='relu')(x)
    x = Dense(encode_size, activation='relu', activity_regularizer=reg, name='Encoding_Layer')(x)

    encoded = x
    x = Dense(layer_sizes[1], activation='relu')(x)
    x = Dense(layer_sizes[0], activation='relu')(x)
    x = Dense(input_size, activation='sigmoid')(x)
    decoded = x

    encoder     = Model(input, encoded)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss=loss)
    return encoder, autoencoder



def encodeData(encoder, data, enc_size):
    encodings = []
    for d in data:
        enc = encoder.predict(d).reshape(-1, enc_size)
        encodings.append(enc)
    return encodings

def modelCnn(nBins=4):
    model = Sequential()
    model.add(Conv2D(10, kernel_size=(3, 3), activation='relu', input_shape=(nBins, nBins, 1)))
    #model.add(Conv2D(20, (3, 3), activation='relu'))
    model.output_shape
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.output_shape
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model

def modelMlp(input_dim=None, nBins=4, nDims=2):
    if input_dim is None: input_dim = nBins ** nDims
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(2, activation='softmax'))
    print(model.output_shape)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model

def modelMlp3(input_dim=None, nBins=4, nDims=2):
    if input_dim is None: input_dim = nBins ** nDims
    model = Sequential()
    model.add(Dense(20, activation='relu', input_dim=input_dim))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    print(model.output_shape)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model

def modelLR(nBins=4):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=nBins * nBins))
    print(model.output_shape)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model

def buildModel(model, X_train, y_train, X_valid, y_valid, batch=128, epochs=10, verbose=1):
    history = History()

    model.fit(X_train, y_train,
              batch_size=batch,
              epochs=epochs,
              verbose=verbose,
              callbacks=[history],
              validation_data=(X_valid, y_valid))

    return history.history['loss']

def ViewEncodings(enc, y, n=10):
    y_ = pa.Series(argmax(y, axis=1))
    y0 = y_[y_==0].index.values
    y1 = y_[y_==1].index.values
    gap = 4
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    def plot(im, p):
        plt.subplot(1, gap + n*2, p)
        im = im.reshape(im.shape[0],1)
        plt.imshow(im, cmap=cmap)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figure(figsize=(20, 8))
    p=1
    for i,yi in enumerate(y0[:n]):
        print (p,yi)
        plot(enc[yi], p)
        p+=1
    p+=gap
    for i, yi in enumerate(y1[:n]):
        plot(enc[yi], p)
        p+=1
    plt.show()



# Look at customers
def Manual():
    d0 = plot1(flag0[0], dfX, df, "Product Group", highlight=targetProducts)
    d1 = plot1(flag1[1], dfX, df, "Product Group", highlight=targetProducts)
    plotCompare(dfX, flag0, flag1, 3)

    dfX.loc['c8553']



# dfXi, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='Index')
# dfXmr, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='MinRng')
# dfXm, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Count', how='Max')


############### MLP - 2Dims #################
n = 10
h=[]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, dims=["F3", "F4"], type='Count', how='Index')
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, dims=["F3", "F4"], type='Count', how='MinRng')
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, dims=["F3", "F4"], type='Count', how='Max')
X_train1, X_valid1, X_test1, y_train1, y_valid1, y_test1 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[])
model1 = modelMlp(nBins=n)
h += buildModel(model1, X_train1, y_train1, X_valid1, y_valid1, epochs=10, batch=5)
plt.plot(h)
myNn.confusion(model1, X_test1, y_test1, mType="MLP")
model1.summary()


############### MLP - 3 Dims #################
n = 10
h=[]
dims=["F2", "F3", "F4"]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, dims = dims, type='Count', how='Max')
X_train1, X_valid1, X_test1, y_train1, y_valid1, y_test1 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[])
model1 = modelMlp(nBins=n, nDims=len(dims))
h += buildModel(model1, X_train1, y_train1, X_valid1, y_valid1, epochs=10, batch=5)
plt.plot(h)
myNn.confusion(model1, X_test1, y_test1, mType="MLP")
model1.summary()







############### Explore #################
import inspect
import time
import random
results = []
done = {0}

def explore(n, dims, how, epochs=10, batch=8, run=0):
    start_time = time.time()
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    paras = {}
    for a in args: paras.update({a: values[a]})
    print ("********************************************************************************")
    print (paras)
    print ("********************************************************************************")
    h=[]
    dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, dims=dims, type='Count', how=how)
    X_train, X_valid, X_test, y_train, y_valid, y_test = myNn.reshapeAndSplitData(dfX, dfY, reshape=[], printing=False)
    model = modelMlp(nBins=n, nDims=len(dims))
    h += buildModel(model, X_train, y_train, X_valid, y_valid, epochs=epochs, batch=batch, verbose =0)
    end_time = time.time()
    duration = end_time - start_time
    cm = myNn.confusion(model, X_test, y_test, show=False)

    print("*** Took {:.2f} seconds".format(duration))
    print()
    stats={}
    stats.update(paras)
    stats.update(cm)
    stats.update({'history':h, 'time':duration})
    return [stats]

def exploreAll(runs = 10, save_every=None, file_pfx="Results", file_path="C:/Temp"):
    results  = []
    ns       = list(range(4,11,2))
    dimss    = list(itertools.combinations(["F1", "F2", "F3", "F4"], 3)) + list(itertools.combinations(["F1", "F2", "F3", "F4"], 2))
    hows     = ['Index', 'MinRng', 'Max']
    epochss  = [10, 15, 20, 25, 30, 40, 50]
    batchs   = [8, 32, 128, 256, 512, 1024]

    def tryArgs():
        n       = random.choice(ns)
        dims    = random.choice(dimss)
        how     = random.choice(hows)
        epochs  = random.choice(epochss)
        batch   = random.choice(batchs)
        args = (n, dims, how, epochs, batch)
        return args

    def nextNewArgs(maxTries=1000):
        for i in range(maxTries):
            args = tryArgs()
            if args not in done:
                done.add(args)
                return args
        return None

    def saveResults(results, run, final=False):
        if final==False:
            if (save_every is None): return
            if (run % save_every): return
        fileName = "{}/{}_{}".format(file_path, file_pfx, run)
        print("*** Saving results to {} ***".format(fileName))
        resultsDf = pa.DataFrame.from_dict(results)
        resultsDf.to_json(fileName+".json")
        resultsDf.to_csv(fileName+".csv")

    combos = len(ns) * len(dimss) * len(hows) * len(epochss) * len(batchs)
    print ("Exploring {} runs out of possible {} combos".format(runs, combos))

    for r in range(runs):
        run = r + 1
        args = nextNewArgs()
        if args is None:
            print("Failed to find new args for run {}".format(run))
            break
        print ("Starting run {} out of {}".format(run, runs))
        results += explore(*args, run=run)
        saveResults(results, run)

    saveResults(results, run, final=True)
    resultsDf = pa.DataFrame.from_dict(results)
    return resultsDf

results += explore(n=4, dims=["F3", "F4"], how='Index', epochs=10, batch=1024)
results += explore(n=12, dims=["F3", "F4"], how='Index', epochs=50, batch=8)
results += explore(n=10, dims=["F3", "F4"], how='Index', epochs=1, batch=8)

results += exploreAll(100, save_every=10)

rdf = pa.read_json("C:/Temp/Results_100.json")
rdf["Dims"] = rdf["dims"].astype('str')

myPlot.compareDistributions(rdf[['Accuracy', 'Recall']], rdf[['how']], showLegend=True)
myPlot.compareDistributions(rdf[['Accuracy', 'Recall']], rdf[['Dims']], showLegend=True)
myPlot.compareDistributions(rdf[['Accuracy', 'Recall']], rdf[['n']], showLegend=True)

myPlot.scatterDf(rdf, ["epochs", "Accuracy"], color=rdf["n"], lblCol='n' )
myPlot.scatterDf(rdf, ["batch", "Accuracy"], color=rdf["n"], lblCol='n' )
myPlot.scatterDf(rdf, ["n", "Accuracy"], color=rdf["n"], lblCol='n' )


myPlot.scatterDf(rdf, ["epochs", "time"], color=rdf["Accuracy"], lblCol='n')
myPlot.scatterDf(rdf, ["batch", "time"], color=rdf["Accuracy"], lblCol='n')



############### CNN #################
n = 10
h=[]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max')
X_train2, X_valid2, X_test2, y_train2, y_valid2, y_test2 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[n,n])
model2 = modelCnn(nBins=n)
h += buildModel(model2, X_train2, y_train2, X_valid2, y_valid2)
plt.plot(h)
myNn.confusion(model2, X_test2, y_test2, mType="CNN")








############### MLP - F1,2,3,4  (1D) #################
# This gives best performance
n = 8
h=[]
xBins, yBins = (n*2, n)
dfX12, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F1", "F2"], omitTarget=True)
dfX34, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F3", "F4"], omitTarget=True)
dfX = pa.concat([dfX12, dfX34], axis = 1)
X_train4, X_valid4, X_test4, y_train4, y_valid4, y_test4 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[])
model4 = modelMlp3(input_dim=xBins*yBins)
h += buildModel(model4, X_train4, y_train4, X_valid4, y_valid4, epochs=25, batch=32)

myNn.confusion(model4, X_train4, y_train4, mType="MLP 3 layers on F1,2,3,4 (Train)")
myNn.confusion(model4, X_test4, y_test4, mType="MLP 3 layers on F1,2,3,4 (Test)")


# => Over-fitting (Test less good)
# Results higher perhaps when didn't omit target products?

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Test) - Omit
# Accuracy  = 76.00% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 72.78% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 77.61% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Train) - Omit
# Accuracy  = 79.48% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 75.46% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 82.15% = Proportion of selected cases that are relevant  = TP / (TP + FP)


# - Confusion Matrix for MLP 3 layers on F1,2,3,4 (Test) - No Omit
# - Accuracy  = 86.17% = Proportion of classifications that are correct  = (TP + TN) / Total
# - Recall    = 89.02% = Proportion of relevant cases that were selected = TP / (TP + FN)
# - Precision = 84.12% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# - Confusion Matrix for MLP 3 layers on F1,2,3,4 (Train) - No Omit
# - Accuracy  = 87.99% = Proportion of classifications that are correct  = (TP + TN) / Total
# - Recall    = 90.29% = Proportion of relevant cases that were selected = TP / (TP + FN)
# - Precision = 86.38% = Proportion of selected cases that are relevant  = TP / (TP + FP)






############### MLP + AE - F1,2,3,4  #################
# Get decent results if use enough epochs in the AE
n = 10
h=[]
xBins, yBins = n*2, n
enc_size = 100
xBins, yBins = n*2, n
dfX12, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F1", "F2"])
dfX34, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F3", "F4"])


dfX = pa.concat([dfX12, dfX34], axis = 1)
# Use reshape=[yBins * xBins] if using ae4b, which is just 1 dimensional
X_train5, X_valid5, X_test5, y_train5, y_valid5, y_test5 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[yBins, xBins])


# Create autoencoder
# Need to run for 50+ epochs.  Loss slowly drops, but decoded image is much sharper
encoder5_1, autoencoder5_1 = ae2(yBins=yBins, xBins=xBins, encode_size=enc_size)
h += buildModel(autoencoder5_1, X_train5, X_train5, X_valid5, X_valid5, epochs=50)
plt.plot(h)
X_train5e, X_valid5e, X_test5e = encodeData(encoder5_1, [X_train5, X_valid5, X_test5], enc_size)

# See how well can recreate inputs
x_decoded = autoencoder5_1.predict(X_test5)
myPlot.PlotImagePairs(X_test5, x_decoded, xBins=xBins, yBins=yBins)

# Clearer max inputs when using ae2
encoder5_1.summary()
images = myNn.find_max_input(encoder5_1, 'dense_354', units=range(0,25))
myPlot.display_images(images, reshape=(yBins,xBins), margin = 1)

# Look at encoding to be used for model
ViewEncodings(X_test5e, y_train5)



model5 = modelMlp3(input_dim=(enc_size))
h += buildModel(model5, X_train5e, y_train5, X_valid5e, y_valid5, epochs=25, batch=5)
myNn.confusion(model5, X_train5e, y_train5, mType="MLP 3 layers on F1,2,3,5 (Train)")
myNn.confusion(model5, X_test5e, y_test5, mType="MLP 3 layers on F1,2,3,5 (Test)")

# Less over-fitting

# Confusion Matrix for MLP 3 layers on F1,2,3,5 (Test) - Omit
# Accuracy  = 69.35% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 78.26% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 66.26% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for MLP 3 layers on F1,2,3,5 (Train) - Omit
# Accuracy  = 71.68% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 79.98% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 68.67% = Proportion of selected cases that are relevant  = TP / (TP + FP)



# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Test) - No Omit
# Accuracy  = 71.27% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 83.99% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 66.40% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Train) - No Omit
# Accuracy  = 71.83% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 84.24% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 67.48% = Proportion of selected cases that are relevant  = TP / (TP + FP)







############### MLP + AE - F1,2,3,4  (OLD) #################
n = 10
h=[]
xBins, yBins = n*2, n
enc_size = 100
xBins, yBins = n*2, n
#dfX12, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F1", "F2"])
#dfX34, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F3", "F4"])
dfX12, dfY = pre_process_data_OLD(df, 'CustIx', nBins=n, type='Pct', dims=["F1", "F2"])
dfX34, dfY = pre_process_data_OLD(df, 'CustIx', nBins=n, type='Pct', dims=["F3", "F4"])

dfX = pa.concat([dfX12, dfX34], axis = 1)
# Use reshape=[yBins * xBins] if using ae4b, which is just 1 dimensional
X_train5, X_valid5, X_test5, y_train5, y_valid5, y_test5 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[yBins, xBins])
nInput = X_train5.shape[1]
print(nInput)

# Create autoencoder
#encoder5_1, autoencoder5_1 = ae5b(nInput, encode_size=enc_size, layer_sizes=[180, 150], pen=None, loss=keras.losses.binary_crossentropy())
# Need to run for 50+ epochs.  Loss slowly drops, but decoded image is much sharper
encoder5_1, autoencoder5_1 = ae2(yBins=yBins, xBins=xBins, encode_size=enc_size)
h += buildModel(autoencoder5_1, X_train5, X_train5, X_valid5, X_valid5, epochs=10)
plt.plot(h)
x_decoded = autoencoder5_1.predict(X_test5)
myPlot.PlotImagePairs(X_test5, x_decoded, xBins=xBins, yBins=yBins)


X_train5e, X_valid5e, X_test5e = encodeData(encoder5_1, [X_train5, X_valid5, X_test5], enc_size)
ViewEncodings(X_test5e, y_train5)
# Clearer max inputs when using ae2
encoder5_1.summary()
images = myNn.find_max_input(encoder5_1, 'dense_326', units=range(0,25))
myPlot.display_images(images, reshape=(yBins,xBins), margin = 1)

model5 = modelMlp3(input_dim=(enc_size))
h += buildModel(model5, X_train5e, y_train5, X_valid5e, y_valid5, epochs=25, batch=5)
myNn.confusion(model5, X_train5e, y_train5, mType="MLP 3 layers on F1,2,3,5 (Train)")
myNn.confusion(model5, X_test5e, y_test5, mType="MLP 3 layers on F1,2,3,5 (Test)")

# Less over-fitting

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Test) - No Omit
# Accuracy  = 71.27% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 83.99% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 66.40% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Train) - No Omit
# Accuracy  = 71.83% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 84.24% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 67.48% = Proportion of selected cases that are relevant  = TP / (TP + FP)













################ With 2D AE ##################
# I think results were using OLD pre-process, and no Omit
n = 10
enc_size=40
h=[]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max')
X_train3, X_valid3, X_test3, y_train3, y_valid3, y_test3 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[n,n], yCat=True)

# Conv AE
encoder3, autoencoder3 = ae3(nBins=n, encode_size=enc_size)
h += buildModel(autoencoder3, X_train3, X_train3, X_valid3, X_valid3, epochs=10)
x_decoded = autoencoder3.predict(X_test3)
myPlot.PlotImagePairs(X_test3, x_decoded)
X_train3e, X_valid3e, X_test3e = encodeData(encoder3, [X_train3, X_valid3, X_test3], enc_size)
ViewEncodings(X_test3e, y_train3)
encoder3.summary()
images = myNn.find_max_input(encoder3, 'Encoding_Layer', units=range(1,20))
myPlot.display_images(images, reshape=(10,10), margin = 1)
print(X_train3e.shape)

model3 = modelMlp3(input_dim=enc_size)
h+=buildModel(model3, X_train3e, y_train3, X_valid3e, y_valid3, epochs=5, batch = 5)
plt.plot(h)
myNn.confusion(model3, X_test3e, y_test3, mType="AE + MLP")

# Confusion Matrix for AE2 + MLP
# Accuracy  = 69.09% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 69.85% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 68.05% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for AE3 + MLP
# Accuracy  = 67.47% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 77.76% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 63.86% = Proportion of selected cases that are relevant  = TP / (TP + FP)





################ With AE F1,2,3,4 ##################
# I think results were using OLD pre-process, and no Omit
n = 6
enc_size=100
h=[]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Count', how='Max', dims=["F2","F3","F4"])
nInput = dfX.shape[1]
X_train3, X_valid3, X_test3, y_train3, y_valid3, y_test3 = myNn.reshapeAndSplitData(dfX, dfY, reshape=[])
X_train3.shape


encoder3, autoencoder3 = ae4b(nInput, encode_size=enc_size, layer_sizes=[200, 150], pen=10e-7)
h += buildModel(autoencoder3, X_train3, X_train3, X_valid3, X_valid3, epochs=20)
plt.plot(h)
x_decoded = autoencoder3.predict(X_test3)
myPlot.PlotImagePairs(X_test3, x_decoded)
X_train3e, X_valid3e, X_test3e = encodeData(encoder3, [X_train3, X_valid3, X_test3], enc_size)
ViewEncodings(X_test3e, y_train3)
encoder3.summary()
images = myNn.find_max_input(encoder3, 'Encoding_Layer', units=range(1,20))
myPlot.display_images(images, reshape=(6,36), margin = 1)


model3 = modelMlp3(enc_size)
h+=buildModel(model3, X_train3e, y_train3, X_valid3e, y_valid3, epochs=5, batch = 5)
plt.plot(h)
myNn.confusion(model3, X_test3e, y_test3, mType="AE + MLP")

# Confusion Matrix for AE2 + MLP
# Accuracy  = 69.09% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 69.85% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 68.05% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for AE3 + MLP
# Accuracy  = 67.47% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 77.76% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 63.86% = Proportion of selected cases that are relevant  = TP / (TP + FP)




