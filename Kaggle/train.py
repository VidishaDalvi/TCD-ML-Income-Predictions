import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option('display.max_rows',100)

file_path1 = 'C:/TCD/DS/Machine Learning/Kaggle/Assign1_income_predictions/tcd ml 2019-20 income prediction training (with labels).csv/train.csv'
train = pd.read_csv(file_path1,na_values={"YearofRecord":["#N/A"],"Gender":["unknown","#N/A","0"],"Age":["#N/A"],"Profession":["#N/A"],"UniversityDegree":["#N/A","0"],"HairColor":["#N/A","0","Unknown"]})
train=train.fillna(0)
train['YearofRecord']=train['YearofRecord'].astype(int)
train['Age']=train['Age'].astype(int)
#print(train.head(100))
file_path2 ='C:/TCD/DS/Machine Learning/Kaggle/Assign1_income_predictions/tcd ml 2019-20 income prediction test (without labels).csv/test.csv'
test = pd.read_csv(file_path1,na_values={"YearofRecord":["#N/A"],"Gender":["unknown","#N/A","0"],"Age":["#N/A"],"Profession":["#N/A"],"UniversityDegree":["#N/A","0"],"HairColor":["#N/A","0","Unknown"]})
test=test.fillna(0)
test['YearofRecord']=test['YearofRecord'].astype(int)
test['Age']=test['Age'].astype(int)
#print(test.head(100))


X_train=train['YearofRecord','Gender','Age','SizeofCity']
y_train=train['Income']
X_test=test['YearofRecord','Gender','Age','SizeofCity']
#y_test=test

le = LabelEncoder()
le.fit(X_train.astype(str))
X_train = le.transform(X_train.astype(str))

le.fit(y_train.astype(str))
y_train = le.transform(y_train.astype(str))

# le = LabelEncoder()
# le.fit(X_train['Gender'].astype(str))
# X_train['Gender'] = le.transform(X_train['Gender'].astype(str))
# X_test['Gender'] = le.transform(X_test['Gender'].astype(str))
#
# le.fit(X_train['SizeofCity'].astype(str))
# X_train['SizeofCity'] = le.transform(X_train['SizeofCity'].astype(str))
# X_test['SizeofCity'] = le.transform(X_test['SizeofCity'].astype(str))
#
# le.fit(X_train['Age'].astype(str))
# X_train['Age'] = le.transform(X_train['Age'].astype(str))
# X_test['Age'] = le.transform(X_test['Age'].astype(str))
#
# le.fit(X_train['Height'].astype(str))
# X_train['Height'] = le.transform(X_train['Height'].astype(str))
# X_test['Height'] = le.transform(X_test['Height'].astype(str))


regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)
plt.scatter(y_test,y_pred)
# print("\n Coefficients="+regr.coef_)
# print("\nMean Squared error="+mean_squared_error(y_test,y_pred))
# print("\n Variance score="+r2_score(y_test,y_pred))
#
# plt.scatter(X_test,y_test,edgecolors="black")
# plt.plot(X_test,y_pred,color="blue",linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
plt.show()