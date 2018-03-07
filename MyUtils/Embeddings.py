
import pandas as pa
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import MyUtils.utils_plot as MyPlot
import MyUtils.utils_prep as MyPrep



def CreateFromArrays(arrays, ofVar = "Item"): 
    model = Word2Vec(arrays, min_count=1)
    return Embeddings(model, ofVar)

def CreateFromDf(df, byVar, ofVar): 
    cols = df.filter([byVar, ofVar])
    gps = MyPrep.groupArrays(cols, byVar, ofVar)
    arrays = gps[ofVar].values
    return CreateFromArrays(arrays, ofVar)



class Embeddings():

    @staticmethod
    def ModelToDf(model, lbl="Item"):
        items = list(model.wv.vocab)
        X = model[model.wv.vocab]
        pca = PCA(n_components=4)
        points = pca.fit_transform(X)
        emb = pa.DataFrame(points, columns=["F1", "F2", "F3", "F4"])
        emb[lbl] = items
        return emb

    def __init__(self, model, itemLbl="Item"):
        self.model = model
        self.df = Embeddings.ModelToDf(model, itemLbl)
        self.lookups = {}
        self.itemLbl = itemLbl
            

    def plotAll(self, show=True): 
        model = self.model
        words = list(model.wv.vocab)
        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        MyPlot.plot_scatter(result, list(model.wv.vocab))
        if show: plt.show()
        return model


    def getGroupData(self, gpLbl, wantGps):
        itemLbl = self.itemLbl
        lu = self.lookups[gpLbl]
        wanted = lu[lu[gpLbl].isin(wantGps)]
        return pa.merge(self.df, wanted , how='inner', on=itemLbl)


    def plotGroups(self, gpLbl, wantGps, dims=["F1", "F2"], highlight = None,  show=True):
        itemLbl = self.itemLbl
        plotData = self.getGroupData(gpLbl, wantGps)
        MyPlot.scatterDf(plotData, dims, itemLbl, color='Blue')
        if highlight is not None:
            MyPlot.scatterDf(self.df.loc[highlight], dims, color='Red')
        if show: plt.show()
        return plotData


    def plotGroup(self, gpLbl, wantGp, dims=["F1", "F2"], highlight=None, show=True):
        self.plotGroups(gpLbl, [wantGp], dims, highlight, show)

    def pairGroups(self, gpLbl, wantGps):
        plotData = self.getGroupData(gpLbl, wantGps)
        sns.pairplot(plotData, hue=gpLbl)
        plt.show()


    def heatGroup(self, gpLbl, wantGp, dims=["F1", "F2"], pct=True):
        itemLbl = self.itemLbl
        gpData = self.getGroupData(gpLbl, [wantGp])
        if pct: MyPlot.heatPlot2Df(gpData, self.df, dims)
        else: MyPlot.heatPlotDf(gpData, dims)
        plt.show()



    def dispGroup(self, gpLbl, wantGp, dims=["F1", "F2"]):
        #gpLbl, wantGp, dims ="CUSTID", 2270000028346, ["F1", "F2"]
        itemLbl = self.itemLbl
        gpData = self.getGroupData(gpLbl, [wantGp])

        # Row 1 = F1 v F2
        plt.subplot(1,2,1)
        MyPlot.scatterDf(gpData, dims, itemLbl)
        plt.subplot(1,2,2)
        MyPlot.heatPlot2Df(gpData, self.df, dims)

        plt.show()


    def dispGroups(self, gpLbl, wantGps):
        itemLbl = self.itemLbl
        f12, f34 = ["F1","F2"], ["F3","F4"]
        n=0
        for g in wantGps:
            n=n+1
            plt.figure(n)
            gpData = self.getGroupData(gpLbl, [g])

            # Row 1 = F1 v F2
            plt.subplot(2,2,1)
            MyPlot.scatterDf(gpData, f12, itemLbl)
            plt.subplot(2,2,2)
            MyPlot.heatPlot2Df(gpData, self.df, f12)

            # Row 2 = F3 v F4
            plt.subplot(2,2,3)
            MyPlot.scatterDf(gpData, f34, itemLbl)
            plt.subplot(2,2,4)
            MyPlot.heatPlot2Df(gpData, self.df, f34)

            plt.show()





    def addLookup(self,gpLbl, lookup):
        self.lookups.update({gpLbl:lookup})

    def getEmbedding(self, item):
        self.model[item]

