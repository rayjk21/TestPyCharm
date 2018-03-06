import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pa
import seaborn as sns
import itertools




# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))                                  




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, show=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if show: plt.show()



def plotPoint(x,y,color='Red'):
    plt.scatter(x, y, color=color)


def plot_scatter(points, labels, xyLbls = ["X","Y"] ,color='Blue', size=20, cmap=None, fSize=10):
    """
        colour can be an array of values in which case the colour map is used
        Otherwise the fixed colour is used
    """
    if cmap == None:cmap=matplotlib.cm.OrRd 

    plt.scatter(points[:, 0], points[:, 1], c=color, s=size, cmap=cmap)

    for i, word in enumerate(labels): 
        plt.annotate(word, xy=(points[i, 0], points[i, 1]), size=fSize)

    plt.xlabel(xyLbls[0])
    plt.ylabel(xyLbls[1])


def scatterDf(df, xyCols, lblCol=None, color='Blue', size=20, cmap=None, fSize=10):
    if lblCol==None : lbls=[]
    else: lbls=list(df[lblCol])
    plot_scatter(df[xyCols].values, lbls, xyCols, color=color, size=size,cmap=cmap, fSize=fSize)


def heatMap(df, xCol="X", yCol="Y", show_legend=True, show_axisName = True, show_axisLbls = True, cmap=None, show=True, diverge=False):
    ras = df.as_matrix()
    min = np.min(ras)
    max = np.max(ras)
    if ((min<0) & (max>0)) | diverge :
        if min==0:min=-0.0001
        if max==0:max=0.0001
        norm = MidpointNormalize(midpoint=0,vmin=min, vmax=max)
        cmap=matplotlib.cm.RdBu_r 
        #cmap = sns.diverging_palette(240, 10, n=9)
        clim=(min, max)
    else:
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        norm = None
        clim = None

    ax = plt.imshow(ras, cmap=cmap, clim=clim, norm=norm, origin='lower')
    #ax = sns.heatmap(df, cmap=cmap, cbar=show_legend)
    #ax.invert_yaxis()


    ax = plt.gca()
 #   plt.xlabel(xCol)
 #   plt.ylabel(yCol)

    if show_axisName:
        ax.set_ylabel(yCol)
        ax.set_xlabel(xCol)

    # Turn off tick labels
    if not(show_axisLbls):
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if show: plt.show()


def heatPlotDf(df,xyCols,xBins=10,yBins=10, labels=False, show=True):
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    xLabels, yLabels = None, None
    if (not labels):
        xLabels = range(xBins) 
        yLabels = range(yBins)
    xCol,yCol = xyCols[0], xyCols[1] 
    x=pa.cut(df[xCol], bins=xBins, labels=xLabels)
    y=pa.cut(df[yCol], bins=yBins, labels=yLabels)
    piv = pa.crosstab(y,x)
    heatMap(piv,xCol,yCol, show=show)



def heatPlot2Df(df1, df2, xyCols, xBins=10,yBins=10, labels=False):
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    # Create bin boundaries
    xCol,yCol = xyCols[0], xyCols[1] 
    xCuts=pa.cut(df2[xCol], bins=xBins, retbins=True)
    yCuts=pa.cut(df2[yCol], bins=yBins, retbins=True)

    xLabels, yLabels = None, None
    if (not labels):
        xLabels = range(xBins) 
        yLabels = range(yBins)

    def getPiv(df):
        x=pa.cut(df[xCol], bins=xCuts[1], labels=xLabels)
        y=pa.cut(df[yCol], bins=yCuts[1], labels=yLabels)
        piv = pa.crosstab(y,x)
        return piv

    p1 = getPiv(df1)
    p2 = getPiv(df2)

    piv = p1 / p2
    ax = sns.heatmap(piv, cmap=cmap)
    ax.invert_yaxis()
    plt.xlabel(xCol)
    plt.ylabel(yCol)


