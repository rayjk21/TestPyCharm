

import matplotlib.pyplot as plt

# For each unique value of waiting_time, plot the % of 'show_up'

from collections import Counter
plt.figure(figsize=(18, 10))

for x in raw_data['waiting_time'].unique():
    count = Counter(raw_data[raw_data['waiting_time'] == x]['show_up'])
    pct =  count[0] / (count[0]+count[1])
    #print(x,pct)
    plt.scatter(x,pct, c='black', s=50)

plt.show()











import seaborn as sns

# Freq plot of categories
sns.countplot(x='show_up', data=raw_data, palette='Set1')
sns.countplot(x='appointment_dayofweek', data=raw_data, palette='GnBu_r')
plt.show()


# Distribution
sns.distplot(raw_data['age'])
# Distribution comparison by 'show_up'
sns.violinplot(x='show_up', y='age', data=raw_data, palette='BuGn_r')


# Multiple plots
fig, ax = plt.subplots(2, 3, figsize=(15, 12))
sns.countplot(x='show_up', data=raw_data, hue='scholarship', ax=ax[0, 0], palette='Set2')
sns.countplot(x='show_up', data=raw_data, hue='hypertension', ax=ax[0, 1], palette='Set2')
sns.countplot(x='show_up', data=raw_data, hue='diabetic', ax=ax[0, 2], palette='Set2')
sns.countplot(x='show_up', data=raw_data, hue='alcoholic', ax=ax[1, 0], palette='Set2')
sns.countplot(x='show_up', data=raw_data, hue='handicap', ax=ax[1, 1], palette='Set2')
sns.countplot(x='show_up', data=raw_data, hue='SMS_received', ax=ax[1, 2], palette='Set2')
plt.show()




#sns.boxplot(x="day", y="total_bill", hue="sex", data=raw, palette="PRGn")
sns.boxplot(y="Cost", x="Product", data=raw, palette="PRGn")
sns.boxplot(x="Cost", y="Destination", data=raw, palette="PRGn")
plt.show()



sns.barplot(x='group', y='Values', data=df, estimator=lambda x: sum(x==0)*100.0/len(x))









# Stacked bar plot
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

#Read in data & create total column
stacked_bar_data = pd.read_csv("C:\stacked_bar.csv")
stacked_bar_data["total"] = stacked_bar_data.Series1 + stacked_bar_data.Series2

#Set general plot properties
sns.set_style("white")
sns.set_context({"figure.figsize": (24, 10)})

#Plot 1 - background - "total" (top) series
sns.barplot(x = stacked_bar_data.Group, y = stacked_bar_data.total, color = "red")

#Plot 2 - overlay - "bottom" series
bottom_plot = sns.barplot(x = stacked_bar_data.Group, y = stacked_bar_data.Series1, color = "#0000A3")


topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})
l.draw_frame(False)

#Optional code - Make plot look nicer
sns.despine(left=True)
bottom_plot.set_ylabel("Y-axis label")
bottom_plot.set_xlabel("X-axis label")

#Set fonts to consistent 16pt size
for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
    item.set_fontsize(16)







# Coloured scatter plots
import seaborn as sns
import pandas as pd
import numpy as np
np.random.seed(1974)

df = pd.DataFrame(
    np.random.normal(10, 1, 30).reshape(10, 3),
    index=pd.date_range('2010-01-01', freq='M', periods=10),
    columns=('one', 'two', 'three'))
df['key1'] = (4, 4, 4, 6, 6, 6, 8, 8, 8, 8)

# Can create matrix of all x vars vs all y vars
sns.pairplot(x_vars=["one"], y_vars=["two"], data=df, hue="key1", size=5)



# Use the 'hue' argument to provide a factor variable
sns.lmplot( x="sepal_length", y="sepal_width", data=df, fit_reg=False, hue='species', legend=False)



# Labelling (doesn't quite work)
plot = sns.regplot( x="F1", y="F2", data=df, fit_reg=False)
for line in range(0,df.shape[0]):
     plot.text(df["F1"][line]+0.2, df["F2"][line], df["Product"][line], horizontalalignment='left', size='medium', color='black', weight='semibold')







################### Centre colour bar

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# create an array of random vlues - you might read in a raster dataset
x=25
y=25
ras=np.random.randint(-1000,3000,size=(x*y)).reshape(x,y)
cmap=matplotlib.cm.RdBu_r # set the colormap to soemthing diverging
plt.imshow(ras, cmap=cmap), plt.colorbar(), plt.show()   


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


elev_min=-1000
elev_max=3000
mid_val=0

mpNorm = MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max)
plt.imshow(ras, cmap=cmap, clim=(elev_min, elev_max), norm=mpNorm)
plt.colorbar()
plt.show()







############## Grid Layout

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