
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import MyUtils.Embeddings as myEmb
import MyUtils.utils_explore as myExp
import MyUtils.utils_prep as myPrep
import MyUtils.utils_plot as myPlot
import MyUtils.utils_nn as myNn

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







#################### Preprocess data - standardise the inputs as Pct or normalised
def pre_process_data(df, byVar, dims=["F3", "F4"], nBins=4, type='Pct', ofVar=None, labels=False, offset='Min') -> pa.DataFrame:
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

dfX, dfY = pre_process_data(df, 'CustIx', nBins=5, type='Pct')

myExp.overview(dfX)
myExp.detail(dfX)

myExp.overview(dfY)




###################### Plot individual member who is/isnt in the target
def plot1(mId, dfX, df, lbl, dims=["F3", "F4"], colorVar=None, highlight=[], show=True, hideAxes=False, r=1, c=2, p1=1, p2=2):
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
    xMin = np.min(df[xDim])
    xMax = np.max(df[xDim])
    yMin = np.min(df[yDim])
    yMax = np.max(df[yDim])

    if colorVar != None : c = mData[colorVar].values
    else: color='Blue'
    #myPlot.scatterDf(mData, dims, lbl, color=color, size=20, fSize=6, hideAxes=True)
    myPlot.scatterDf(mData, dims, lbl, color=color, size=20, fSize=6, hideAxes=hideAxes, xRange=(xMin,xMax), yRange=(yMin,yMax))
    highlightPoints()
    plt.title(title)

    # Heatmap of model input
    plt.subplot(r,c,p2)
    myPlot.heatMap(dfX.loc[mId], show_legend=True, show=False, diverge=True, hideAxes=hideAxes)

    if show: plt.show()

def plotN(custs, highlight=[]):
    n=len(custs)
    for i,cust in enumerate(custs):
        print("{} {}".format(i,cust))
        plot1(cust, dfX, df, "Product Group", show=False, r=2,c=n,p1=i+1,p2=i+1+n, highlight=highlight, hideAxes=True)
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



plot1('c1', dfX, df, "Product Group", highlight=targetProducts)

plot1(flag0[0], dfX, df, "Product Group", highlight=targetProducts)
plot1(flag1[0], dfX, df, "Product Group", highlight=targetProducts, dims=["F1", "F2"])
plot1(flag1[0], dfX, df, "Product Group", highlight=targetProducts, dims=["F3", "F4"])

plotN(flag1[10:15], highlight=targetProducts)

plotCompare(dfX, flag0, flag1, nGrid=5)





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
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return encoder, autoencoder

def ae2(nBins=10, xBins=None, yBins=None, encode_size=25):
    xBins, yBins = myNn.setBins(nBins, xBins, yBins)
    input = Input(shape=(yBins, xBins, 1))
    x = Conv2D(8,  (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(1,  (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(encode_size*2, activation='relu')(x)
    x = Dense(encode_size, activation='relu')(x)
    encoded = x

    x = Dense(encode_size*2, activation='relu')(x)
    x = Dense(yBins*xBins, activation='relu')(x)
    x = Reshape((yBins,xBins,1))(x)
    x = Conv2D(8 , (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = x

    encoder     = Model(input, encoded)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
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
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
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
    return model

def modelMlp(input_dim=None, nBins=4):
    if input_dim is None: input_dim = nBins * nBins
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(2, activation='softmax'))
    print(model.output_shape)
    return model

def modelMlp3(input_dim=None, nBins=4,xBins=None, yBins=None):
    xBins, yBins = myNn.setBins(nBins, xBins, yBins)

    if input_dim is None: input_dim = xBins * yBins
    model = Sequential()
    model.add(Dense(20, activation='relu', input_dim=input_dim))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    print(model.output_shape)
    return model

def modelLR(nBins=4):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=nBins * nBins))
    print(model.output_shape)
    return model

def buildModel(model, X_train, y_train, X_valid, y_valid, useCat=False, batch=128, epochs=10):
    opt = keras.optimizers.Adadelta()
    if useCat:
        print("Using categorical_crossentropy")
        loss=keras.losses.categorical_crossentropy
    else:
        print("Using binary_crossentropy")
        loss=keras.losses.binary_crossentropy

    history = History()

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=batch,
              epochs=epochs,
              verbose=1,
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



############### MLP #################
n = 10
h=[]
#dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Norm', dims=["F3", "F4"])
#dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct', dims=["F3", "F4"], offset='Mean')
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct', dims=["F3", "F4"], offset='Min')

X_train1, X_valid1, X_test1, y_train1, y_valid1, y_test1 = myNn.reshapeAndSplitData(dfX, dfY, '1D', nBins=n)
model1 = modelMlp(nBins=n)
h += buildModel(model1, X_train1, y_train1, X_valid1, y_valid1, epochs=10, batch=5)
plt.plot(h)
myNn.confusion(model1, X_test1, y_test1, mType="MLP")




############### CNN #################
n = 10
h=[]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct')
X_train2, X_valid2, X_test2, y_train2, y_valid2, y_test2 = myNn.reshapeAndSplitData(dfX, dfY, '2D', nBins=n)
model2 = modelCnn(nBins=n)
h += buildModel(model2, X_train2, y_train2, X_valid2, y_valid2)
plt.plot(h)
myNn.confusion(model2, X_test2, y_test2, mType="CNN")




################ With AE ##################
n = 10
enc_size=40
h=[]
dfX, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct')
X_train3, X_valid3, X_test3, y_train3, y_valid3, y_test3 = myNn.reshapeAndSplitData(dfX, dfY, '2D', nBins=n, yCat=True)

encoder3, autoencoder3 = ae3(nBins=n, encode_size=enc_size)
h += buildModel(autoencoder3, X_train3, X_train3, X_valid3, X_valid3, epochs=10, useCat=False)
x_decoded = autoencoder3.predict(X_test3)
myNn.PlotImagePairs(X_test3, x_decoded)
X_train3e, X_valid3e, X_test3e = encodeData(encoder3, [X_train3, X_valid3, X_test3], enc_size)
ViewEncodings(X_test3e, y_train3)
encoder3.summary()

imgages = myNn.find_max_input(encoder3, 'Encoding_Layer', units=range(0,10))

n = len(imgages)
for i,img in enumerate(imgages):
    plt.subplot(1,n,i+1)
    img = img/255
    img = img.reshape(10,10)
    plt.imshow(img)


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










############### MLP - F1,2,3,4  #################
n = 10
xBins, yBins = n*2, n
dfX12, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct', dims=["F1", "F2"])
dfX34, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct', dims=["F3", "F4"])
dfX = pa.concat([dfX12, dfX34], axis = 1)
X_train4, X_valid4, X_test4, y_train4, y_valid4, y_test4 = myNn.reshapeAndSplitData(dfX, dfY, '1D', xBins=xBins, yBins=yBins)
model4 = modelMlp3(xBins=xBins, yBins=yBins)
buildModel(model4, X_train4, y_train4, X_valid4, y_valid4, epochs=5, batch=32)
myNn.confusion(model4, X_train4, y_train4, mType="MLP 3 layers on F1,2,3,4 (Train)")
myNn.confusion(model4, X_test4, y_test4, mType="MLP 3 layers on F1,2,3,4 (Test)")

# => Over-fitting (Test less good)

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Test)
# Accuracy  = 77.92% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 74.90% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 79.05% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Train)
# Accuracy  = 81.24% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 78.96% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 82.73% = Proportion of selected cases that are relevant  = TP / (TP + FP)






############### MLP + AE - F1,2,3,4  #################
n = 10
h=[]
xBins, yBins = n*2, n
enc_size = 40
dfX12, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct', dims=["F1", "F2"])
dfX34, dfY = pre_process_data(df, 'CustIx', nBins=n, type='Pct', dims=["F3", "F4"])
dfX = pa.concat([dfX12, dfX34], axis = 1)
X_train4, X_valid4, X_test4, y_train4, y_valid4, y_test4 = myNn.reshapeAndSplitData(dfX, dfY, '2D', xBins=xBins, yBins=yBins)

encoder4_2, autoencoder4_2 = ae2(xBins=xBins, yBins=yBins, encode_size=enc_size)
buildModel(autoencoder4_2, X_train4, X_train4, X_valid4, X_valid4, epochs=25)
myNn.PlotImagePairs(X_test4, autoencoder4_2.predict(X_test4), xBins=xBins, yBins=yBins)
# Encode
X_train4e, X_valid4e, X_test4e = encodeData(encoder4_2, [X_train4, X_valid4, X_test4], enc_size)
ViewEncodings(X_test4e, y_train4)

encoder4_3, autoencoder4_3 = ae3(xBins=xBins, yBins=yBins, encode_size=enc_size)
buildModel(autoencoder4_3, X_train4, X_train4, X_valid4, X_valid4, epochs=10)
myNn.PlotImagePairs(X_test4, autoencoder4_3.predict(X_test4), xBins=xBins, yBins=yBins)
# Encode
X_train4e, X_valid4e, X_test4e = encodeData(encoder4_3, [X_train4, X_valid4, X_test4], enc_size)
ViewEncodings(X_test4e, y_train4)

model4 = modelMlp3(input_dim=(enc_size))
h += buildModel(model4, X_train4e, y_train4, X_valid4e, y_valid4, epochs=5, batch=5)
myNn.confusion(model4, X_train4e, y_train4, mType="MLP 3 layers on F1,2,3,4 (Train)")
myNn.confusion(model4, X_test4e, y_test4, mType="MLP 3 layers on F1,2,3,4 (Test)")

# Less over-fitting

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Test)
# Accuracy  = 71.27% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 83.99% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 66.40% = Proportion of selected cases that are relevant  = TP / (TP + FP)

# Confusion Matrix for MLP 3 layers on F1,2,3,4 (Train)
# Accuracy  = 71.83% = Proportion of classifications that are correct  = (TP + TN) / Total
# Recall    = 84.24% = Proportion of relevant cases that were selected = TP / (TP + FN)
# Precision = 67.48% = Proportion of selected cases that are relevant  = TP / (TP + FP)















def confusion(model, X_test, y_test, yCat=True, show=True, mType=""):
    """
        return y_test, y_pred

        model, X_test, y_test = model1, X_test1, y_test1
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
    print("Confusion Matrix for " + mType)
    print("Accuracy  = {:.2%} = {}".format(acc, accLbl))
    print("Recall    = {:.2%} = {}".format(rec, recLbl))
    print("Precision = {:.2%} = {}".format(pre, preLbl))
    print()
    myPlot.plot_confusion_matrix(cmNN,[0,1], show=show, title=mType+" Confusion Matrix")
