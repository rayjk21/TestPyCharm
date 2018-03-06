
import pandas as pa
import numpy as np

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import MyUtils.utils_plot as MyPlot
import MyUtils.utils_prep as MyPrep




def embeddingsToDf(model, lbl="Item", gpMap=None):
    items = list(model.wv.vocab)
    X = model[model.wv.vocab]
    pca = PCA(n_components=4)
    points = pca.fit_transform(X)
    emb = pa.DataFrame(points, columns=["F1", "F2", "F3", "F4"])
    emb[lbl] = items

    if gpMap != None:
        gps = np.array(list(map(lambda item: gpMap[item], items)))
        emb["Group"]=gps

    return emb


def plotEmbeddings(model): 
    words = list(model.wv.vocab)
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    MyPlot.plot_scatter(result, list(model.wv.vocab))
    return model






def embeddings(arrays): 
    model = Word2Vec(arrays, min_count=1)
    plotEmbeddings(model)
    return model



def embeddingsFromDf(df, byVar, ofVar): 
    cols = df.filter([byVar, ofVar])
    gps = MyPrep.groupArrays(cols, byVar, ofVar)
    arrays = gps[ofVar].values
    return embeddings(arrays)
