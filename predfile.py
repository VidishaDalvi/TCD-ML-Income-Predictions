import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option('display.max_rows',100)


data_train = pd.read_csv("C:/TCD/DS/Machine Learning/Kaggle/Assign1_income_predictions/tcd ml 2019-20 income prediction training (with labels).csv/train.csv")
data_test = pd.read_csv("C:/TCD/DS/Machine Learning/Kaggle/Assign1_income_predictions/tcd ml 2019-20 income prediction test (without labels).csv/test.csv")

data_train = data_train.rename(index=str, columns={"Income in EUR" : "Income"})

data = pd.concat([data_train, data_test], sort=False)
data.head()
data = data.drop("Instance", axis=1)
data = data.rename(columns={'Body Height [cm]':"Height"})
data = data.rename(index=str, columns={"Year of Record" : "YearOfRecord"})
data = data.rename(index=str, columns={"Size of City" : "SizeOfCity"})
data = data.rename(index=str, columns={"University Degree": "UniversityDegree"})
data = data.rename(index=str, columns={"Wears Glasses" : "WearsGlasses"})
data = data.rename(index=str, columns={"Hair Color" : "HairColor"})
data = data.rename(index=str, columns={"Income in EUR" : "Income"})

len = len(data_train)

data['Gender'] = data['Gender'].replace('0', "other")
data['Gender'] = data['Gender'].replace('unknown', pd.np.nan)

data.YearOfRecord.unique()
data.Gender.unique()
data.Age.unique()
data.UniversityDegree.unique()

data['UniversityDegree'] = data['UniversityDegree'].replace('PhD', 4)
data['UniversityDegree'] = data['UniversityDegree'].replace('Master', 3)
data['UniversityDegree'] = data['UniversityDegree'].replace('Bachelor', 2)
data['UniversityDegree'] = data['UniversityDegree'].replace('No', 0)
data['UniversityDegree'] = data['UniversityDegree'].replace(pd.np.nan, 0)

data.WearsGlasses.unique()

data.HairColor.unique()
data['HairColor'] = data['HairColor'].replace('0', pd.np.nan)
data['HairColor'] = data['HairColor'].replace('Unknown', pd.np.nan)

def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)

data['Country'] = calc_smooth_mean(data, 'Country', 'Income', 2)

data['Profession'] = calc_smooth_mean(data, 'Profession', 'Income', 50)


data.drop("YearOfRecord", axis=1)
data.drop("Country", axis=1)
data.drop("WearsGlasses", axis=1)

data = pd.get_dummies(data, columns=["Gender"], drop_first = True)
data = pd.get_dummies(data, columns=["HairColor"], drop_first = True)


Xtrain = data[0:len]
Xtrain["YearOfRecord"].fillna((Xtrain["YearOfRecord"].mean()), inplace=True )
Xtrain["Age"].fillna((Xtrain["Age"].mean()), inplace=True )
Xtrain["Profession"].fillna((Xtrain["Profession"].mean()), inplace=True )

Ytrain = Xtrain[["Income"]]

Xtrain = Xtrain.drop("Income", axis=1)

#del data


X_training, X_holdOut, Y_training, Y_holdOut = train_test_split(Xtrain, Ytrain, train_size=0.9, random_state=100)
LR = RandomForestRegressor(n_estimators= 100, random_state=100)
model = LR.fit(X_training, Y_training)
ypred = model.predict(X_holdOut)

mse = mean_squared_error(Y_holdOut, ypred)
print(mse)
rmse = math.sqrt(mse)
print(rmse)


X_test = data[len:]
X_test = X_test.drop("Income", axis=1)
X_test["YearOfRecord"].fillna((X_test["YearOfRecord"].mean()), inplace=True )
X_test["Age"].fillna((X_test["Age"].mean()), inplace=True )
X_test["Profession"].fillna((X_test["Profession"].mean()), inplace=True )
X_test["Country"].fillna((X_test["Country"].mean()), inplace=True )


pred = model.predict(X_test)
pred = pd.DataFrame(pred)

print(pred.head())
pred.to_csv("C:/TCD/DS/Machine Learning/Kaggle/Assign1_income_predictions/submission5.csv", sep=',', index=False, header=True)