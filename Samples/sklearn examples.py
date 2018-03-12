


# Prepare datasets
X = raw_data.drop(['show_up'], axis=1)
y = raw_data['show_up']
Counter(y)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)
Counter(y_res)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101) # 80:20
Counter(y_train)
Counter(y_test)





# Encoding
from sklearn.preprocessing import LabelEncoder

encoder_neighbourhood = LabelEncoder()
raw_data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(raw_data['neighbourhood'])
raw_data['neighbourhood_enc'].value_counts()
le = LabelEncoder()
prods = raw["Product Group"].astype('str').unique()
le.fit(prods)
le.classes_
le.transform(['VITAMINS'])




# Use scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
raw['Year0'] = MinMaxScaler().fit_transform(raw[['Year']])
raw['Cost0'] = StandardScaler().fit_transform(raw[['Cost']])






# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
clf.feature_importances_
clf.score(X_test, y_test)



from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))