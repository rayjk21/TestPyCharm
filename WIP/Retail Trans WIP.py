
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import MyUtils.Embeddings as myEmb
import MyUtils.utils_explore as myExp
import MyUtils.utils_prep as myPrep
import MyUtils.utils_plot as myPlot

import sklearn.preprocessing as skPrep

pa.set_option('max_rows', 11)
pa.set_option('expand_frame_repr', False)

###################################################################################
# Retail
###################################################################################

rRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Retail\All Cust Items Data Grid.txt", sep='\t')
rRaw["Department"].astype(str)
rRaw["Product Group"].astype(str)
rRaw['Product'] = "_" + rRaw['Product Group'].astype(str)

myExp.overview(rRaw)
myExp.detail(rRaw)

# Drop very small/large customers and baskets
productsPerBasket  = myPrep.xPerY(rRaw,"PRODUCT", "BASKET ID","PperB" ).where(lambda x: (x>1) & (x<50)).dropna()
basketsPerCustomer = myPrep.xPerY(rRaw,"BASKET ID","CUSTID", "BperC" ).where(lambda x: (x>2) & (x<20)).dropna()

#productsPerBasket.plot.hist(range(0,50,5))
#basketsPerCustomer.plot.hist(range(0,20,2))

df1 = pa.merge(rRaw, basketsPerCustomer.to_frame(), how='inner', left_on=["CUSTID"], right_index=True)
df2 = pa.merge(df1, productsPerBasket.to_frame(),  how='inner', left_on=["BASKET ID"], right_index=True)

dfSubSet = df2

myExp.overview(df1)
myExp.overview(df2)







###################### Create feature vectors
df = dfSubSet
deptEmb = myEmb.CreateFromDf (df, 'CUSTID', 'Department')
#deptEmb.plotAll()

pgpEmb = myEmb.CreateFromDf (df, 'CUSTID', 'Product')
df3 = pa.merge(df, pgpEmb.df, on='Product', how='inner')
df3 = df3.set_index("CUSTID")

dfFeature = df3





############# Investigate Target Products
productPerDept = myPrep.xPerY(rRaw,'Product', 'Department' )
productToDept = rRaw[['Product', 'Department']].drop_duplicates()
pgpEmb.addLookup ("Department", productToDept)

chinese = pgpEmb.df["Product"].str.contains("CHINESE")
indian  = pgpEmb.df["Product"].str.contains("INDIAN")
cou     = pgpEmb.df["Product"].str.contains("COU ")
target = (chinese | indian) & (~cou)
targetProducts = list(pgpEmb.df.loc[target]["Product"])

uDepts = list(productPerDept.index.values)

# Oriental is identified more by F3 & F4
pgpEmb.plotGroup("Department","MEAL SOLUTIONS", ["F1","F2"], highlight=target)
pgpEmb.plotGroup("Department","MEAL SOLUTIONS", ["F3","F4"], highlight=target)

pgpEmb.plotGroups("Department",uDepts, ["F3","F4"], highlight=target)








################ Preprocess Y data Apply Target flag to all purchases
df = dfFeature
targetPurchases  = df[df["Product"].isin(targetProducts)]
custBuyingTarget = targetPurchases.index.drop_duplicates().to_frame()
custBuyingTarget["Target"] = True

df4 = pa.merge(df, custBuyingTarget, right_index=True, left_index=True, how='left')
df4["Target"] = df4["Target"].fillna(False)
df4["Target"].value_counts()


dfY = df4["Target"]
flag0 = np.unique(dfY.where(lambda x: x==False).dropna().index.values)
flag1 = np.unique(dfY.where(lambda x: x==True).dropna().index.values)
allCusts = basketsPerCustomer.index.values




#################### Preprocess X data - standardise the inputs as Pct or normalised
df=dfFeature
def pre_process_X(df, byVar, dims=["F3", "F4"], nBins=4, type='Pct', ofVar=None) -> pa.DataFrame:
    """
    Preprocess data for Neural Network
        nBins = Number of bins for each of the dimensions
        type = 'Norm' to use normalised ratings
        type = 'Pct'  to use the percentage of ratings

        :rtype:
            Pandas dataframe
            Returns 2D array (shape nBins,nBins) for each byVar
    """

    def calcPct(df, byVar, dims, nBins=5, labels=False):
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
        xCol, yCol = dims

        xCuts = pa.cut(df[xCol], bins=nBins, retbins=True)[1]
        yCuts = pa.cut(df[yCol], bins=nBins, retbins=True)[1]

        xLabels, yLabels = None, None
        if (not labels):
            xLabels = range(nBins)
            yLabels = range(nBins)

        df["X"] = pa.cut(df[xCol], bins=xCuts, labels=xLabels).astype(str)
        df["Y"] = pa.cut(df[yCol], bins=yCuts, labels=yLabels).astype(str)
        df["Freq"] = 1

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

        # Standardise, setting to 0.5 if there is only one value (hence range==0)
        eachPctRng[eachPctRng==0] = 1.0
        eachStd = ((pctM - eachPctMin) / eachPctRng).reshape(nCust,nBins,nBins)
        eachStd[eachStd==0] = 0.5
        np.nan_to_num(eachStd, copy=False)

 #      i = uCust.get_loc(1000000004055)
 #      mat[i,:]
 #      eachPctRng[i]
 #      eachPctMin[i]
 #      eachPctMax[i]
 #      eachPctSum[i]
 #      eachPctCount[i]
 #      eachStd[i]

        stds = eachStd.reshape(nCust * nBins, nBins)
        custIx = np.repeat(uCust, nBins)
        cols = [byVar] + list(xLabels)
        dfX = pa.DataFrame(np.column_stack([custIx, stds]),columns=cols)
        dfX = dfX.set_index(byVar)
        return dfX

    def calcNorm(df, byVar, ofVar, dims, nBins=5, labels=False):
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
        xCol, yCol = dims

        xCuts = pa.cut(df[xCol], bins=nBins, retbins=True)[1]
        yCuts = pa.cut(df[yCol], bins=nBins, retbins=True)[1]

        xLabels, yLabels = None, None
        if (not labels):
            xLabels = range(nBins)
            yLabels = range(nBins)

        df["X"] = pa.cut(df[xCol], bins=xCuts, labels=xLabels).astype(str)
        df["Y"] = pa.cut(df[yCol], bins=yCuts, labels=yLabels).astype(str)

        allMean = df[ofVar].mean()
        allStd = df[ofVar].std()
        cellMeans = pa.pivot_table(df, index=[byVar, "Y"], columns=["X"], values=ofVar, aggfunc=np.mean, dropna=False)
        cellMeans = cellMeans.fillna(allMean)
        cellNorms = (cellMeans - allMean) / allStd

        return cellNorms

    if type == 'Pct':
        dfX = calcPct(df, byVar, dims, nBins)

    if type == 'Norm':
        dfX = calcNorm(df, byVar, ofVar, dims, nBins)

    return dfX

dfX = pre_process_X(df, 'CUSTID', nBins=5, type='Pct')






###################### Plot individual member who does or doesn't like the film
def plot1(mId, dfX, df, lbl, dims=["F3", "F4"], colorVar=None, highlight=[], show=True, r=1, c=2, p1=1, p2=2):
    """
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
    xMin = min(df[xDim])
    xMax = max(df[xDim])
    yMin = min(df[yDim])
    yMax = max(df[yDim])

    if colorVar != None : c = mData[colorVar].values
    else: color='Blue'
    #myPlot.scatterDf(mData, dims, lbl, color=color, size=20, fSize=6, hideAxes=True)
    myPlot.scatterDf(mData, dims, lbl, color=color, size=20, fSize=6, hideAxes=True, xRange=(xMin,xMax), yRange=(yMin,yMax))
    highlightPoints()
    plt.title(title)

    # Heatmap of model input
    plt.subplot(r,c,p2)
    myPlot.heatMap(dfX.loc[mId], show_legend=True, show=False, diverge=True, hideAxes=True)

    if show: plt.show()

def plotN(custs, highlight=[]):
    n=len(custs)
    for i,cust in enumerate(custs):
        print("{} {}".format(i,cust))
        plot1(cust, dfX, df, "Product", show=False, r=2,c=n,p1=i+1,p2=i+1+n, highlight=highlight)
    plt.show()

# Display comparison heat plots
def plotCompare(dfX, flag0, flag1, nGrid=3):
    nn = nGrid
    p = 0
    for i in range(nn):
        for j in range(nn):
            plt.subplot2grid((nn, nn * 2), (i, j))
            myPlot.heatMap(dfX.loc[flag0[p]], show_legend=False, show_axisName=False, show_axisLbls=False, show=False,
                           diverge=True)
            plt.subplot2grid((nn, nn * 2), (i, j + nn))
            myPlot.heatMap(dfX.loc[flag1[p]], show_legend=False, show_axisName=False, show_axisLbls=False, show=False,
                           diverge=True)
            p = p + 1
    plt.show()


plot1(flag0[0], dfX, df, "Product", highlight=targetProducts)
plot1(flag1[0], dfX, df, "Product", highlight=targetProducts)

plotN(flag1[10:15], highlight=targetProducts)

plotCompare(dfX, flag0, flag1)