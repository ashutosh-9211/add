# -*- coding: utf-8 -*-
'''OM SHRI GANESHAY NAMAH'''
""" using a powerful model which depends upon Gradient Boosted Decision Tree"""
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae# : for evaluation of maodel
data= pd.read_csv('melb_data.csv')
#print(data.describe()) to just get a confirmation and overview of loaded data
#print(data.columns) : to get look on columns in data set
data= data.dropna(axis=0) #:to drop missing values

def xgboostModel(train_X,test_X,train_y,test_y):
    from xgboost import XGBRegressor as xgb
    model= xgb(n_estimators=1000, learning_rate=0.05)
    model.fit(train_X,train_y,early_stopping_rounds=5,eval_set=[(test_X, test_y)], verbose=False)
    pred= model.predict(test_X)
    return mae(pred,test_y)
    

"""choosing features for predictions"""
y= data.Price  #: choosing output features
input_feature=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X= data[input_feature]
#print(X.head()): to check if data is loaded or not


'''Splitting the data to get more accurate model'''
from sklearn.model_selection import train_test_split as tts
train_X,test_X,train_y,test_y= tts(X,y,random_state=0)

print('%2f',xgboostModel(train_X,test_X,train_y,test_y))