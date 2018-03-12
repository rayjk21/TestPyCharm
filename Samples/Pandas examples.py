
import pandas as pa
import pandas as pd
import numpy as np



pa.set_option('max_rows', 9)
pa.set_option('expand_frame_repr', False)



# Input data
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
df = pa.Series(list('abcaa'))
df = pd.DataFrame([[1,2], [3,4]], columns=['a', 'b'])

raw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Movies\Movies_2m.csv", encoding = "ISO-8859-1")



# create a sample of OPs unique values
series = pd.Series(np.random.randint(low=0, high=3, size=100))
mapper = {0: 'New York', 1: 'London', 2: 'Zurich'}
nomvar = series.replace(mapper)

pa.get_dummies(df, prefix='col')


# Creating
a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1) # From series
  




# Summary of data frame
df.info
df.head()
df.sample(10)
df.describe()









# Summary Stats of SERIES
raw['waiting_time'].value_counts()
raw["Destination"].unique()



# Aggregations
group='Product'
df.groupby(group).groups.keys()

df.groupby(group).count()                   # Creates dataframe
df.groupby(['Year', 'Product']).count()     # Creates dataframe with hierarchical index
df.groupby(group)['Cost'].mean()            # Creates series

df.groupby(['Year']).agg({'Cost':[min, max, 'mean'], 'Product':'count'}) # Creates df

# The columns have a multi-level index that you can collapse with
diffs.columns = ["_".join(x) for x in diffs.columns.ravel()]


piv = pa.pivot_table(df, values="Cost",index=["Year"], columns=["Product"], fill_value=0)
piv = pa.pivot_table(raw, values="Cost",index=["Year"], columns=["Destination"], fill_value=0)

# Use dropna to ensure all R & C present
cellMeans = pa.pivot_table(df, index=[byVar,"Y"], columns=["X"], values=ofVar, aggfunc=np.mean, fill_value=0, dropna=False)







# Plots
plt.hist(df.Cost, bins=5, facecolor='red', alpha=0.5, label="Cost")
plt.legend()
plt.show()


agg = df.groupby('Year')['Cost'].mean()
agg.plot.bar()
plt.show()



piv = pa.pivot_table(df, values="Cost",index=["Year"], columns=["Product"], fill_value=0)
sns.heatmap(piv)
plt.show()


piv2 = pa.pivot_table(df, values="Cost",index=["Year"], columns=["Destination"], fill_value=0)
sns.heatmap(piv2)
plt.show()








# Columns
raw.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)


raw = raw.rename(columns={'Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 
                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 
                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 
                                    'Handcap': 'handicap', 'No-show': 'show_up'})

dests = raw.filter(['Person URN', 'Destination'])





# Data Types
raw = raw.select_dtypes(['int64']).apply(pd.Series.astype, dtype='category')


raw['Income'] = raw['Income'].astype('category')
raw['Occupation'] = raw['Occupation'].astype('category')
raw['age'] = raw['age'].astype('int64')
raw.info()






# Values
raw['sex'] = raw['sex'].map({'F': 0, 'M': 1})
raw['show_up'] = raw['show_up'].map({'No': 1, 'Yes': 0})
raw['scheduled_day'] = pd.to_datetime(raw['scheduled_day'], infer_datetime_format=True)
raw['appointment_day'] = pd.to_datetime(raw['appointment_day'], infer_datetime_format=True)
raw['waiting_time'] = list(map(lambda x: x.days, raw['appointment_day'] - raw['scheduled_day']))
raw['waiting_time'] = raw['waiting_time'].apply(lambda x: 1 if(x > 1) else 0)
raw['appointment_dayofweek'] = raw['appointment_day'].map(lambda x: x.dayofweek)





# Rows
raw.drop(raw[raw['waiting_time'] < -1].index, inplace=True)
df = raw.head(20)





# Indexing rows (the condition returns the index values for the main df)
dests = raw.loc[raw["Destination"]=="Italy"]
dests = raw.loc[raw["Destination"].isin(["Italy", "France"])]

df.loc[(df['column_name'] == some_value) & df['other_column'].isin(some_values)]
df.loc[df['column_name'] != some_value]
df.loc[~df['column_name'].isin(some_values)]


# Dataframe uses mask, which sets non selected rows to Nan
df.where(df["A"]=='a')
df.where(df["A"]=='a').dropna()   # Hide these rows

# df indexing applies mask and drops Nan
df[df["A"]=='a']
df[(df["A"]=='a') & (df["B"]=='b')]
df[df["A"]=='a' & df["B"]=='b']  # Type cast error without brackets

# df.loc is equivalent to mask
df.loc[df["A"]=='a']
df.loc[[True,False,True], 'C']     # Can also specify column

# Update values with given row and column
df.loc[[True,False,True], 'C'] = 9






# Filter rows (pass bool array of which rows you want)
raw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Movies\Movies_200k.csv")
def inGenre(df, genre):
    dGenres = df["Genre"]
    getGenre = lambda multi : multi.split("|")
    isGenre  = lambda gList : genre in gList
    bGenres = dGenres.apply(getGenre).apply(isGenre)
    return df[bGenres.values]

inGenre(raw, "Western")



# Dedup rows
wanted = inGenre(topVol, "Western")[["Title", "Genre"]].drop_duplicates()





# GroupBy (is slow)
urns = dests.groupby('Person URN')

i=10
for urn,gp in urns:
    if i>0:
        print (urn)
        print (gp["Destination"])
    i=i-1




# Convert to ndarray
df = pd.DataFrame([[1,2], [3,4],[5,6]], columns=['a', 'b'])
a=df.iloc[:,1:]
b=df.iloc[:,1:].values

print(type(df))
print(type(a))
print(type(b))

c = df.as_matrix()
print(type(c))
print(c)
print(c.shape)






# Joining on Columns
movieEmb = embeddingsToDf(movieMod, "Title")
showMovies = inGenre(topVol, "Western")
pa.merge(movieEmb, showMovies, how='inner', on='Title')


# Joining on Index
movieEmb = embeddingsToDf(movieMod, "Title").set_index("Title")
showMovies = inGenre(topVol, "Western").set_index("Title")
pa.merge(movieEmb, showMovies, how='inner', left_index=True, right_index=True)











######################################################################################
# SERIES
######################################################################################


# Series: filtering
raw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Movies\Movies_200k.csv")

# value_counts() returns a series
# where returns full length series but with NaN for the value
# dropna removes the rows with NaN, ie that fail the where
raw["Title"].value_counts().where(lambda x: x==601).dropna()

# Dont seem to be able to have multiple conditions in the lambda
# - so define m, and use in the where to index the series
m = topVol["Member ID"].value_counts()
m.where(m>10).where(m<50).dropna()




raw["Rating"].value_counts().sort_values()
raw["Rating"].value_counts().sort_index()

# Filter the dataframe
raw[raw["Rating"] >= 4.0]





# Banding a series
r=pa.cut(df[row], bins=rBins, labels=rLabels)
c=pa.cut(df[col], bins=cBins, labels=cLabels)
piv = pa.crosstab(r,c)






# Apply function to each value of series
def inGenre(genre):
    dGenres = raw["Genre"]
    getGenre = lambda ar : ar.split("|")
    sGenres = dGenres.apply(getGenre)
    return sGenres

list(inGenre("Action")[0:10])





# Set same flag to all memeber records, based on one title's rating
def FlagRatingsOver(df, title, over=3.0):
    ixUnder = (df["Rating"] < over)
    ixOver  = (df["Rating"] >= over)
    ixTitle = (df["Title"] == title)
    col = "Like_" + title
    df.loc[ixTitle & ixUnder, col] = 0
    df.loc[ixTitle & ixOver,  col] = 1
    tData = df[["Member ID", col]].dropna().drop_duplicates()
    flag=pa.merge(df[["Member ID"]], tData, on="Member ID", how='inner')
    df[col] = flag
    print(df[col].value_counts())




# Combine Series
s1 = pd.Series([1, 2], index=['A', 'B'], name='s1')
s2 = pd.Series([3, 4], index=['A', 'B'], name='s2')
pd.concat([s1, s2], axis=1)
pd.concat([s1, s2], axis=1).reset_index()
