# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:54:40 2020

@author: Gauri
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("startups.csv")
X = df.iloc[:, : -1].values
Y = df.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

X = np.array(X)
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X) #.toarray()


X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

from statsmodels import api as sm
#BACKWARD ELIMINATION WITH S.L = 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()