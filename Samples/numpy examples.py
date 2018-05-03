



array_np = numpy.asarray(array)
low_values_flags = array_np < lowValY  # Where values are low
array_np[low_values_flags] = 0  # All low values set to 0



# assign zero to all elements less than or equal to `lowValY`
a[a<=lowValY] = 0 
# find n-th largest element in the array (where n=highCountX)
x = partial_sort(a, highCountX, reverse=True)[:highCountX][-1]
# 
a[a<x] = 0 #NOTE: it might leave more than highCountX non-zero elements
           # . if there are duplicates





np.count_nonzero(y_pred[y_pred>=cut])





# Create array of arrays for a grouped column
df = pa.DataFrame( {'a':np.random.randint(0,60,600), 'b':[1,2,5,5,4,6]*100})

def groupArrays(df, byVar):
         keys,values=df.sort_values(byVar).values.T
         ukeys,index=np.unique(keys,True)
         arrays=np.split(values,index[1:])
         df2=pa.DataFrame({'a':ukeys,'b':[list(a) for a in arrays]})
         return df2

groupArrays(df, 'a')






# Apply function (cant get to work for a 1D array)
x = np.array((('aa.txt',0),('b.tct',0)))
s = np.apply_along_axis(lambda a: (a[0].split('|')[0]),1,x)
print (x)
print (s)

rRaw["CustIx"] = np.vectorize(lambda x: 'c'+str(x))(le.transform(rRaw["CUSTID"]))



# Formatting

float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})




# Extend shape to 3D
out.shape = a.shape + (ncols,)



# Concatenating
lol = [[1,4,5,8,9,10], [2,3,4,7,8,9], [5,10,15]]
def np2(lol):
    return np.array([np.array(i) for i in lol])

lol
np.hstack(lol)
np.ravel(lol)

ts_ = np2(lol)
np.hstack(ts_)




# Vectorising

# Lambda gets applied to each element in array 1,2,3
offset = (lambda x: 1000-x)
offset(5)
offsetV = my_prep.vectorizeA(offset)
offsetV(np.array([1,2,3]))

# In principle this lambda works...
offset = (lambda x: pa.Timestamp.now() - np.timedelta64(x,'D'))
offset(5)

# But when vectorised the 'x' is actually a np.array of size 1
offset = (lambda x: pa.Timestamp.now() - np.timedelta64(np.asscalar(x),'D'))
offsetV = my_prep.vectorizeA(offset)
offsetV(np.array([1,2,3]))

