# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:01:37 2021

@author: 313
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import neighbors, svm, tree, ensemble, linear_model, neural_network, naive_bayes
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_excel(r'E:\1 PredictSadMusic\data\data_2.xlsx')#数据   

y = data.loc[:,['Preference']]
y = y.values
x = data.iloc[:, 3:] #x = data.iloc[:, 3:] #all; x = data.iloc[:, 3:12] #individual only; x = data.iloc[:, 13:] #audio only
x = x.values

df_result = pd.DataFrame()
model = ensemble.RandomForestRegressor(n_estimators= 99, max_depth=18, random_state=20, min_samples_leaf=23, 
                                       min_samples_split=5, max_features = 0.6 )
#model = linear_model.LinearRegression()  #线性回归
split_n = 10
kf = KFold(n_splits = split_n,shuffle = True,random_state = None)
aa = 0
bb = 0
df_result = pd.DataFrame()
df_importance = pd.DataFrame()
for train_index, test_index in kf.split(x):                
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index] 
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    aa = aa + r2
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    bb = bb + rmse
    list_combine = [[r2,]]
    df1 = pd.DataFrame(list_combine)
    df_result = df_result.append(df1)
    print(r2)
    importance = model.feature_importances_
    df_imp =pd.DataFrame(importance)
    df_importance = pd.concat([df_importance, df_imp], axis= 1)
r2_mean = aa/split_n  
rmse_mean = bb/split_n          
print('r2_mean = {}'.format(r2_mean)) 
print('rmse_mean = {}'.format(rmse_mean))            
df_result.to_csv('result.csv') 
df_importance.to_csv('feature_importances.csv')   