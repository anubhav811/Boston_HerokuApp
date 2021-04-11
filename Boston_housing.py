# BOSTON HOUSING PRICES 


## Importing the required libraries and the given dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('housing.csv')

## Adding feature names

feature_names= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'N', 'LSTAT', 'MEDV']
dataset = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=feature_names)
# MEDV is the target value(dependent variable)

## Train-Test Splitting

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


## Scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


## Applying Linear Regression Model

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

## Saving the model

from joblib import dump, load
dump(model, 'Boston.joblib') 

## Using the model

from joblib import dump, load
import numpy as np
model = load('Boston.joblib') 





