



raw = pa.get_dummies(raw, columns=['Income'], prefix='Income')
raw = pa.get_dummies(raw, columns=['Occupation'], prefix='Occupation')










# Prepare datasets
from collections import Counter
X = raw_data.drop(['show_up'], axis=1)
y = raw_data['show_up']
Counter(y)


# Get sample sample size of y=0 and y=1
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)
Counter(y_res)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101)






# Create standardized pivot table
def calcNorm(df=input, nBins=5, dims = ["F1", "F2"], byVar="Member ID", ofVar="Rating", labels = False):
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

    allMean   = df[ofVar].mean()
    allStd    = df[ofVar].std()
    cellMeans = pa.pivot_table(df, index=[byVar,"Y"], columns=["X"], values=ofVar, aggfunc=np.mean, dropna=False)
    cellMeans = cellMeans.fillna(allMean)
    cellNorms = (cellMeans - allMean) / allStd

    return norm