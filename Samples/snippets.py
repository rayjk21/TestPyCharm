

# Standardise values for each byVar
def calcMean(df=input, nBins=5, dims = ["F1", "F2"], byVar="Member ID", ofVar="Rating", labels = False):
    """
    df=input
    nBins=4
    dims = ["F1", "F2"]
    byVar="Member ID"
    ofVar="Rating"
    xCol, yCol = dims
    labels = False
    """
    
    xCuts=pa.cut(input[xCol], bins=nBins, retbins=True)[1]
    yCuts=pa.cut(input[yCol], bins=nBins, retbins=True)[1]

    xLabels, yLabels = None, None
    if (not labels):
        xLabels = range(nBins) 
        yLabels = range(nBins)

    df["X"] = pa.cut(df[xCol], bins=xCuts, labels=xLabels).astype(str)
    df["Y"] = pa.cut(df[yCol], bins=yCuts, labels=yLabels).astype(str)


    cellMeans = pa.pivot_table(df, index=[byVar,"Y"], columns=["X"], values=ofVar, aggfunc=np.mean, dropna=False)
    eachMean  = pa.pivot_table(df, index=[byVar], values=ofVar, aggfunc=np.mean)
    eachStd   = pa.pivot_table(df, index=[byVar], values=ofVar, aggfunc=np.std)

    cellMeans.loc[12]
    eachMean.loc[12]

    ix1 = cellMeans.index.get_level_values(0).drop_duplicates()
    ix2 = eachMean.index.get_level_values(0)
    nGps = len(ix1)

    meanXY = cellMeans.as_matrix().reshape(nGps,nBins * nBins)
    meanT  = eachMean.as_matrix()

    cellMeans.loc[12]
    mean[0]

    std = eachStd.as_matrix()
    np.where(std==0)
    cellMeans.loc[ix2[2272]]
    std[2272]

    norm = (meanXY - eachMean.as_matrix()) / (eachStd.as_matrix())

    eachPiv = eachPiv.fillna(value=0)
    return eachPiv
