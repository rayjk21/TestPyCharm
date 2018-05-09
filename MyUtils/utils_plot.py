
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pa
import seaborn as sns
import itertools
import math
import MyUtils.my_stats as myStats
from scipy import stats

import colorsys
import seaborn as sns
import matplotlib





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


def setBins(nBins, xBins=None, yBins=None):
    if xBins is None: xBins = nBins
    if yBins is None: yBins = nBins
    return xBins, yBins


def make2D(img):
    v = img.reshape(1,-1)
    n = v.shape[1]
    nBins = int(math.sqrt(n))
    if (nBins * nBins != n):
        v = v[0,0:(nBins*nBins)]
        print("Cropping shape {} to {}x{}".format(img.shape, nBins, nBins))
    return v.reshape(nBins,nBins)


def __test():
    img = np.array([1,2,3,4,5])
    make2D(img)
    img = np.array([[1, 2], [3, 4]])


def PlotImagePairs(orig, decoded, nBins=None, xBins=None, yBins=None, n=10):
    xBins, yBins = setBins(nBins, xBins, yBins)
    def getShape(img):
        if (yBins is None):
            return make2D(img)
        else:
            return img.reshape(yBins, xBins)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)

        plt.imshow(getShape(orig[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(getShape(decoded[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()





def display_images(images, reshape=None, margin = 5):
    # Calc size of grid
    nImages = len(images)
    n = math.ceil(math.sqrt(nImages))
    if reshape:
        img_width, img_height            = reshape[0:2]
        nChannels = 1
    else:
        img_width, img_height, nChannels = images[0].shape

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched = np.zeros((width, height, nChannels))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            ix = i * n + j
            if (ix>=nImages): break
            if reshape is None:
                img = images[ix]
            else:
                img = images[ix].reshape(img_width, img_height, nChannels)

            stitched[(img_width + margin) * i: (img_width + margin) * i + img_width,
                     (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img


    if nChannels==1:
        stitched = stitched.reshape(width,height)
    print("Image of shape {}".format(stitched.shape))
    plt.imshow(stitched)
    return stitched
    # save the result to disk
    #imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)



def stacked_bar(df, ax=None):
    if ax is None:
        ax = plt.gca()
    # Cumulate so bars don't hide each other
    resultsT = df.sort_index(ascending=False).cumsum(axis=0).fillna(0).sort_index().T
    cmap = sns.color_palette("Set2", len(resultsT.columns))
    hatches = ['|', '/', '.','','-']
    for i, c in enumerate(resultsT.columns):
        hatch = hatches [int(i / 8)]
        ax.bar(resultsT.index.values, resultsT.iloc[:, i], color=cmap[i], label=str(c), hatch = hatch)
        #sns.barplot(resultsT.index.values, resultsT.iloc[:, i], color=cmap[i], label=str(c))

    ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1, 1))


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




def showHideAxes(plt, hide=False):
    if (hide):
        plt.xlabel("")
        plt.ylabel("")

    # Turn off tick labels
    ax = plt.gca()
    if (hide):
        ax.set_yticklabels([])
        ax.set_xticklabels([])


def plot_scatter(points, labels, xyLbls=None, color='Blue', size=20, cmap=None, fSize=10, hideAxes=False, show_axisLbls=True, xRange=None, yRange=None):
    """
        colour can be an array of values in which case the colour map is used
        Otherwise the fixed colour is used
    """
    if xyLbls is None:
        xyLbls = ["X", "Y"]
    if cmap == None:cmap=matplotlib.cm.OrRd

    plt.scatter(points[:, 0], points[:, 1], c=color, s=size, cmap=cmap)

    for i, word in enumerate(labels): 
        plt.annotate(word, xy=(points[i, 0], points[i, 1]), size=fSize)

    if (not hideAxes):
        plt.xlabel(xyLbls[0])
        plt.ylabel(xyLbls[1])


    # Turn off tick labels
    ax = plt.gca()
    if ((not show_axisLbls) | hideAxes):
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if (xRange is not None):
        ax.set_xlim(xRange[0], xRange[1])
    if (yRange is not None):
        ax.set_ylim(yRange[0], yRange[1])


def scatterDf(df, xyCols, lblCol=None, color='Blue', size=20, cmap=None, fSize=10, hideAxes=False, xRange=None, yRange=None):
    if lblCol==None : lbls=[]
    else: lbls=list(df[lblCol])
    plot_scatter(df[xyCols].values, lbls, xyCols, color=color, size=size,cmap=cmap, fSize=fSize, hideAxes=hideAxes, xRange=xRange, yRange=yRange)


def heatMap(arr, xCol="X", yCol="Y", show_legend=True, hideAxes=False, show_axisName = True, show_axisLbls = True, cmap=None, show=True, diverge=False):
    # arr = dfX.loc[flag0[p]]
    # arr.columns
    if (type(arr) is pa.DataFrame):
        arr = arr.as_matrix()

    ras =  arr.astype('float')
    #ras =  df.as_matrix().astype('float')  # Now pass as array
    vmin = np.min(ras)
    vmax = np.max(ras)
    if ((vmin<0) & (vmax>0)) | diverge :
        if vmin==0:vmin=-0.0001
        if vmax==0:vmax=0.0001
        norm = MidpointNormalize(midpoint=0,vmin=min, vmax=vmax)
        cmap=matplotlib.cm.RdBu_r 
        #cmap = sns.diverging_palette(240, 10, n=9)
        clim=(vmin, vmax)
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

    if (show_axisName & (not hideAxes)):
        ax.set_ylabel(yCol)
        ax.set_xlabel(xCol)

    # Turn off tick labels
    if ((not show_axisLbls) | hideAxes):
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if show: plt.show()


def heatPlotDf(df,xyCols,xBins=10,yBins=10, trueLabels=False, hideAxes=False, show=True):
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    xLabels, yLabels = None, None

    if (not trueLabels):
        xLabels = range(xBins) 
        yLabels = range(yBins)
    xCol,yCol = xyCols[0], xyCols[1] 
    x=pa.cut(df[xCol], bins=xBins, labels=xLabels)
    y=pa.cut(df[yCol], bins=yBins, labels=yLabels)
    piv = pa.crosstab(y,x)
    heatMap(piv,xCol,yCol, show=show, hideAxes=hideAxes)



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



def compareDistribution(xSeries, catSeries, ignoreValue = None, title="", showAxes=True, showLegend=True, show=True, cmap=None, nBins=10):
    '''
        Creates single plot, for the xSeries, showing distributioins for each category in the catSeries
    :param xSeries:
    :param catSeries:
    :param ignoreValue:
    :param title:
    :param showAxes:
    :param showLegend:
    :param show:
    :param cmap:
    :return:
    '''
    xmin = np.nanmin(xSeries)
    xmax = np.nanmax(xSeries)
    bins = np.linspace(xmin,xmax, nBins)
    keep = lambda x: x != ignoreValue
    keepV = np.vectorize(keep)
    s01 = None
    xData = []
    if (cmap is None): cmap = matplotlib.cm.get_cmap('Greys')

    def get_x(cat):
        catIx = catSeries[catSeries==cat].dropna().index
        xs = xSeries[catIx].dropna()
        return xs[keepV(xs)]

    cats = list(catSeries.drop_duplicates().values)

    for cat in cats:
        x = get_x(cat)
        plt.hist(x, bins, alpha=0.5, label=str(cat), normed=True)
        xData.append(x)

    if len(cats)==2:
        x0, x1 = (xData[0], xData[1])
        np.mean(x0)
        np.mean(x1)
        p = stats.ttest_ind(x0, x1).pvalue
        s01 = myStats.getSig01(p)
        if s01:
            colour = cmap(s01/2)
            ax = plt.gca()
            ax.set_facecolor(colour)

    showHideAxes(plt, hide = ~showAxes)

    if showLegend: plt.legend(loc='upper right')
    if s01: title = "{}:: {:0.2f}".format(title, s01)
    if not(title==""): plt.title(title, fontsize = 8)
    if show: plt.show()


def compareDistributions(dfX, dfY, ignore=None, showLegend=False, showAxes=False, nBins=10):
    '''
    Creates tiled plot, one plot for each value, showing distributioins for each category


    :param dfX: Each column contains a distribution to be plotted
    :param dfY: Single column where each value indicates the category of that row
    :param ignore:
    :return:
    '''
    catSeries = dfY.iloc[:, 0]
    xIxs = len(dfX.columns)
    sq = math.ceil(math.sqrt(xIxs))
    xIx = 0
    for xIx in range(xIxs):
        plt.subplot(sq, sq, xIx+1)
        xSeries = dfX.iloc[:,xIx]
        xLbl = dfX.columns.ravel()[xIx]
        compareDistribution(xSeries, catSeries, ignoreValue = ignore, title=xLbl, showAxes=showAxes, showLegend=showLegend, show=False)
    plt.suptitle(str(list(dfX.columns.names)))
    plt.show()

