

import itertools
import more_itertools



dict(x[i:i+2] for i in range(0, len(x), 2))





# Convert array of arrays to dictionary
d = {}
for row in pairs:
    d[row[0]] = row[1]  


colours = list(map(lambda prod: d[prod], labels))





# Mapping and Collecting
raw = pa.read_csv(r"C:\VS Projects\Numerics\Numerics\Temp.FSharp\Data\Movies\Movies_200k.csv")

cGenres = topVol["Genre"].value_counts().keys()
def sepGenres(genre): 
    return genre.split("|")

dGenres = itertools.chain.from_iterable( map(sepGenres, cGenres))
uGenres = list(more_itertools.unique_everseen(dGenres))

