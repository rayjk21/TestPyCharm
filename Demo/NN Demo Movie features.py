import numpy as np
import pandas as pa
import seaborn as sns
import sklearn
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib 
from matplotlib import pyplot as plt
import pydot
import graphviz
import itertools
import more_itertools
import MyUtils.utils_explore as MyExp
import MyUtils.utils_prep as MyPrep
import MyUtils.utils_plot as MyPlot
import MyUtils.utils_nn as MyMod
import MyUtils.Embeddings as Emb
import tensorflow
import keras as keras


pa.set_option('max_rows', 11)
pa.set_option('expand_frame_repr', False)


###################################################################################
# Holidays
###################################################################################

hRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Holidays\Bookings for All People.csv")
MyExp.overview(hRaw)
MyExp.detail(hRaw)

destEmb = Emb.CreateFromDf(hRaw,'Person URN', "Destination")
destEmb.plotAll()

destEmb.df





###################################################################################
# Retail
###################################################################################

rRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Retail\All Cust Items Data Grid.txt", sep='\t')
rRaw["Department"].astype(str)
rRaw["Product Group"].astype(str)
rRaw['Product'] = "_" + rRaw['Product Group'].astype(str)


# Create features
deptEmb = Emb.CreateFromDf (rRaw, 'CUSTID', 'Department')
deptEmb.plotAll()
pgpEmb = Emb.CreateFromDf (rRaw, 'CUSTID', 'Product')




# Add Dept & Cust lookups
productToDept = rRaw.filter(['Product', 'Department']).drop_duplicates()
pgpEmb.addLookup ("Department", productToDept)

custLu = rRaw[['CUSTID', 'Product']].drop_duplicates()
pgpEmb.addLookup ("CUSTID", custLu)




# Unique customers & depts
uCust = rRaw["CUSTID"].value_counts().where(lambda x: (x>10) & (x<100)).dropna()
uDepts = list(rRaw['Department'].drop_duplicates())


pgpEmb.dispGroup("CUSTID", "1140000006790")
pgpEmb.dispGroup("CUSTID", "2270000028346")
pgpEmb.dispGroup("CUSTID", "1290000013483")





###################################################################################
# Movies
###################################################################################

mRaw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Movies\Movies_2m.csv", encoding = "ISO-8859-1")
MyExp.overview(mRaw)
MyExp.detail(mRaw)


# Keep ratings of high volume films that were liked
top = mRaw[mRaw["Rating"] >= 4.0]
top["Title"].value_counts()

# Keep hi volume
hiVol = top["Title"].value_counts().where(lambda x: x>=750).dropna()
print(hiVol)

# Keep just high volume & top rating
topVol = top[top["Title"].isin(hiVol.keys())]

MyExp.overview(topVol)
MyExp.detail(topVol)




############### Create embeddings based on high volume films that were liked
movieEmb = Emb.CreateFromDf(topVol, 'Member ID', 'Title')





#### Visualise some films 
f12, f34 = ["F1", "F2"], ["F3", "F4"]

def genrePlot(genre, dims=["F1", "F2"], highlight=None):
    def inGenre(df, genre):
        dGenres = df["Genre"]
        getGenre = lambda multi : multi.split("|")
        isGenre  = lambda gList : genre in gList
        bGenres = dGenres.apply(getGenre).apply(isGenre)
        return df[bGenres.values]

    def plotTitle(fTitle, dims):
        x,y=dims
        tData = movieEmb.df[movieEmb.df["Title"]==fTitle]
        plt.scatter(tData[x], tData[y], color='Red')

    wanted = inGenre(topVol, genre)[["Title", "Genre"]].drop_duplicates()
    showMovies = pa.merge(movieEmb.df, wanted , how='inner', on='Title')
    MyPlot.scatterDf(showMovies, dims, "Title", color='Blue')
    if highlight != None: plotTitle(highlight, dims)
    plt.show()


genrePlot("Comedy", f12, highlight="Dumb & Dumber")
genrePlot("Comedy", f34, highlight="Dumb & Dumber")

# Add Member lookup
memberLu = topVol[['Member ID', 'Title']].drop_duplicates()
movieEmb.addLookup("Member ID", memberLu)

movieEmb.dispGroup("Member ID", 98372 , dims=f34)  # Prefers F3
movieEmb.dispGroup("Member ID", 23812 , dims=f34)  # Prefers F4
movieEmb.dispGroup("Member ID", 117092, dims=f34)  # Prefers F4






############# Preprocess data to highlight film of iterest
fTitle =  "Dumb & Dumber"

# Flag the main dataframe based on whether the member likes "Dumb & Dumber"
def GetFlagsOver(df, title, over=3.0):
    """
    Get flag for each member based on whether their rating 
    for the given film is above the given threshold
    """
    ixUnder = (df["Rating"] < over)
    ixOver  = (df["Rating"] >= over)
    ixTitle = (df["Title"] == title)
    col = "Flag"
    flags=pa.DataFrame(df["Member ID"], columns=["Member ID", col])
    flags.loc[ixTitle & ixUnder, col] = 0
    flags.loc[ixTitle & ixOver,  col] = 1
    return flags.dropna().drop_duplicates()

# Get flag for each member based on whether their rating for the given film
flags = GetFlagsOver(mRaw, fTitle, over=3.5)
# Add flag to the full ratings dataframe - drop if have not rated the film
flagged = pa.merge(mRaw, flags, on='Member ID', how='left').dropna()


# Members have 1 if like, 0 if dislike
print(flags)
print(flagged.loc[flagged["Member ID"]==152])
print(flagged["Flag"].value_counts())




# Add Embeddings to the main dataframe, to give features of each Title
keep = ['Member ID', 'Movie ID', 'Title', 'Rating', 'Flag']
input = pa.merge(flagged[keep], movieEmb.df, on="Title", how='inner')

# Add index and sort by Member ID
input["Member Ix"] = input["Member ID"]
input = input.set_index("Member Ix")
input = input.sort_values(["Member ID"])


# Preview counts
input["Flag"].value_counts()
input["Rating"].value_counts()
print(input.loc[input["Member ID"]==152])
MyExp.detail(input)





# Identify some members with 10-50 Titles who like/dislike 
def uniqueMembers(df, minR=10, maxR=50):
    m = df.index.value_counts()
    df = m.where(m>=minR).where(m<=maxR).dropna().to_frame()
    df.columns=["Count"]
    df["Member ID"] = df.index
    return df

uMembers = pa.merge(uniqueMembers(input, 40, 50), flags, on="Member ID", how='inner')
flag0 = list(uMembers[uMembers["Flag"]==0]["Member ID"].head(50))
flag1 = list(uMembers[uMembers["Flag"]==1]["Member ID"].head(50))




###############  Create Neural Networks #######################

from collections import Counter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report



# Preprocess data for Neural Network - standardise the inputs as Pct or normalised
def preProcessData(df, nBins=4, type='Norm', dims=["F3", "F4"]):
    """
    Preprocess data for Neural Network 
        nBins = Number of bins for each of the dimensions
        type = 'Norm' to use normalised ratings
        type = 'Pct'  to use the percentage of ratings
    """

    def calcPct(df, nBins=5, dims = ["F1", "F2"], byVar="Member ID", labels = False):
        """
        df=input
        nBins=5
        dims = ["F1", "F2"]
        byVar="Member ID"
        xCol, yCol = dims
        labels = False
        """
        xCol, yCol = dims

        xCuts=pa.cut(df[xCol], bins=nBins, retbins=True)[1]
        yCuts=pa.cut(df[yCol], bins=nBins, retbins=True)[1]

        xLabels, yLabels = None, None
        if (not labels):
            xLabels = range(nBins) 
            yLabels = range(nBins)

        df["X"] = pa.cut(df[xCol], bins=xCuts, labels=xLabels)
        df["Y"] = pa.cut(df[yCol], bins=yCuts, labels=yLabels)
        df["Freq"]=1

        eachPiv = pa.pivot_table(df, index=[byVar,"Y"], columns=["X"], values="Freq", aggfunc=np.sum)
        basePiv = pa.pivot_table(df, index=["Y"], columns=["X"], values="Freq", aggfunc=np.sum)

        pct = eachPiv / basePiv
        pct = pct.fillna(value=0)
        return pct

    def calcNorm(df, nBins=5, dims = ["F1", "F2"], byVar="Member ID", ofVar="Rating", labels = False):
        """
        df=input
        nBins=4
        dims = ["F1", "F2"]
        byVar="Member ID"
        ofVar="Rating"
        xCol, yCol = dims
        labels = False
        """
        xCol, yCol = dims
    
        xCuts=pa.cut(df[xCol], bins=nBins, retbins=True)[1]
        yCuts=pa.cut(df[yCol], bins=nBins, retbins=True)[1]

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

        return cellNorms

    if type=='Norm':
        dfX = calcNorm(df=df, nBins=nBins, dims = dims)
    if type=='Pct' :
        dfX = calcPct(df=df, nBins=nBins, dims = dims)

    dfY = df[["Member ID", "Flag"]].drop_duplicates()["Flag"]

    return dfX, dfY

# Display comparison heat plots
def plotCompare(dfX, nGrid=3, flag0=flag0, flag1=flag1):
    nn=nGrid
    p=0
    for i in range(nn):
        for j in range(nn):
            plt.subplot2grid((nn,nn*2),(i,j))
            MyPlot.heatMap(dfX.loc[flag0[p]], show_legend=False, show_axisName = False, show_axisLbls = False, show=False, diverge=True)
            plt.subplot2grid((nn,nn*2),(i,j+nn))
            MyPlot.heatMap(dfX.loc[flag1[p]], show_legend=False, show_axisName = False, show_axisLbls = False, show=False, diverge=True)
            p=p+1
    plt.show()

# Plot individual member who does or doesn't like the film
def plot1(mId, dfX, df=input, dims=["F3", "F4"], lbl="Title", highlight=None):
    """
        df  = Data for each member for each Title
        dfX = X values for modelling, summarised for each member over titles
    """

    def plotTitle():
        tData = mData[mData["Title"]==highlight]
        MyPlot.scatterDf(tData, dims, color='Grey', size=100)

    # Members individual titles
    plt.subplot(1,2,1)
    mData = df.loc[mId]
    flag = mData.head(1)["Flag"].values[0]
    title = ("Member: " + str(mId) + " - Flag: " + str(flag))
    cmap=matplotlib.cm.OrRd 
    if highlight != None: plotTitle()
    MyPlot.scatterDf(mData, dims, lbl, color=mData["Rating"].values, size=20, cmap=cmap, fSize=6)

    # Heatmap of model input
    plt.subplot(1,2,2)
    MyPlot.heatMap(dfX.loc[mId], show_legend=True, show=False, diverge=True)

    plt.suptitle(title)
    plt.show()

def reshapeAndSplitData(dfX, dfY, shape='1D', nBins=4, yCat = True):
    """
    Reshape the data and split into train/valid/test as 60:20:20
    """
    y = dfY.as_matrix()
    nMemb = y.shape[0]

    if shape=='1D':
        X = dfX.as_matrix().reshape(nMemb, nBins * nBins)
    else:
        X = dfX.as_matrix().reshape(nMemb, nBins, nBins, 1)

    X_train, X_, y_train, y_         = train_test_split(X, y,   test_size=0.4, random_state=101) 
    X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, test_size=0.5, random_state=101) 

    if yCat:
        y_test  = keras.utils.to_categorical(y_test, 2)
        y_valid = keras.utils.to_categorical(y_valid, 2)
        y_train = keras.utils.to_categorical(y_train, 2)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

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
    return model

def modelMlp(nBins=4):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=nBins * nBins))
    model.add(Dense(2, activation='softmax'))
    print(model.output_shape)
    return model

def modelLR(nBins=4):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=nBins * nBins))
    print(model.output_shape)
    return model

def buildModel(model, X_train, X_valid, y_train, y_valid, yCat=True, epochs=20):
    opt = keras.optimizers.Adadelta()
    if yCat:  loss=keras.losses.categorical_crossentropy
    else:     loss=keras.losses.binary_crossentropy

    print("Input shape: {}".format(X_train.shape))

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=5,
              epochs=epochs,
              verbose=1,
              validation_data=(X_valid, y_valid))

def confusion(model, X_test, y_test, yCat=True, show=True, mType=""):
    """
        return y_test, y_pred
    """
    if yCat:
        y_pred = model.predict(X_test)[:,1]
        cut=0.5
        y_pred[y_pred>=cut] = 1
        y_pred[y_pred<cut] = 0
        y_test = argmax(y_test, axis=1)
    else:
        y_pred = model.predict(X_test)
        cut=0.5
        y_pred[y_pred>=cut] = 1
        y_pred[y_pred<cut] = 0
        y_test=y_test.astype(int)

    cmNN = confusion_matrix(y_test, y_pred)

    acc = sklearn.metrics.accuracy_score(y_test, y_pred)      # (TP + TN) / Total
    rec = sklearn.metrics.recall_score(y_test, y_pred)        # TP / (TP + FN)
    pre = sklearn.metrics.precision_score(y_test, y_pred)     # TP / (TP + FP)

    accLbl = "Proportion of classifications that are correct  = (TP + TN) / Total"
    recLbl = "Proportion of relevant cases that were selected = TP / (TP + FN)"
    preLbl = "Proportion of selected cases that are relevant  = TP / (TP + FP)"
    print()
    print("Accuracy  = {:.2%} = {}".format(acc, accLbl))
    print("Recall    = {:.2%} = {}".format(rec, recLbl))
    print("Precision = {:.2%} = {}".format(pre, preLbl))
    print()
    MyPlot.plot_confusion_matrix(cmNN,[0,1], show=show, title=mType+" Confusion Matrix")






############### MLP #################
n = 4
dfX, dfY = preProcessData(df=input, nBins=n)
# Look at 2 customers
d0 = plot1(flag0[0], dfX, highlight=fTitle)
d1 = plot1(flag1[1], dfX, highlight=fTitle)
plotCompare(dfX, 3)
X_train1, X_valid1, X_test1, y_train1, y_valid1, y_test1 = reshapeAndSplitData(dfX, dfY, '1D', nBins=n)
model1 = modelMlp(nBins=n)
buildModel(model1, X_train1, X_valid1, y_train1, y_valid1)
confusion(model1, X_test1, y_test1, mType="MLP")

############### CNN #################
n = 10
dfX, dfY = preProcessData(df=input, nBins=n)
plotCompare(dfX, 3)
X_train2, X_valid2, X_test2, y_train2, y_valid2, y_test2 = reshapeAndSplitData(dfX, dfY, '2D', nBins=n)
model2 = modelCnn(nBins=n)
buildModel(model2, X_train2, X_valid2, y_train2, y_valid2)
confusion(model2, X_test2, y_test2, mType="CNN")


############### Logistic Regression #################
n = 10
dfX, dfY = preProcessData(df=input, nBins=n)
plotCompare(dfX, 3)
X_train3, X_valid3, X_test3, y_train3, y_valid3, y_test3 = reshapeAndSplitData(dfX, dfY, '1D', nBins=n, yCat=False)
model3 = modelLR(nBins=n)
buildModel(model3, X_train3, X_valid3, y_train3, y_valid3, yCat=False)
confusion(model3, X_test3, y_test3, yCat=False, mType="Logistic Regression")


