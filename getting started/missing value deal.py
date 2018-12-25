# -*- coding: utf-8 -*-
''' OM SHRI GANESHAY NAMAH'''

''' Dealing with missing value : there are two major methods 
    1. The Simple One: dropping the missing value containing column
    2. Simple Imputer: filling missing values with the mean value
'''
' Dropping the missing value column'
def drop(data):
    column_with_missing_values= [col for col in data.columns if data[col].isnull().any()]
    data=data.drop(column_with_missing_values,axis=1)
    return data

'Using Simple Imputer'
def imputation(data):
    from sklearn.preprocessing import Imputer as si
    my_imputer= si()
    data=my_imputer.fit_transform(data)
    return(data)
    
def rfrModel(train_X,test_X,train_y,test_y):
    """using random forest reegressor"""
    from sklearn.ensemble import RandomForestRegressor as rfr
    model= rfr(random_state=1)# defining the model
    model.fit(train_X,train_y)# fitting the model on training sets
    y_pred= model.predict(test_X)#predicting on test values
    '''Evaluating your model using mean absolute error'''
    return mae(test_y,y_pred)

import pandas as pd
from sklearn.metrics import mean_absolute_error as mae# : for evaluation of maodel
data= pd.read_csv('melb_data.csv')
#print(data.describe()) to just get a confirmation and overview of loaded data
#print(data.columns) : to get look on columns in data set
data= data.dropna(axis=0) #:to drop missing values
data=drop(data)

"""choosing features for predictions"""
y= data.Price  #: choosing output features
input_feature=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X= data[input_feature]
#print(X.head()): to check if data is loaded or not


'''Splitting the data to get more accurate model'''
from sklearn.model_selection import train_test_split as tts
train_X,test_X,train_y,test_y= tts(X,y,random_state=0)
print(rfrModel(train_X,test_X,train_y,test_y))


"""choosing features for predictions"""
y= data.Price  #: choosing output features
input_feature=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X= imputation(data[input_feature])
#print(X.head()): to check if data is loaded or not


'''Splitting the data to get more accurate model'''

train_X,test_X,train_y,test_y= tts(X,y,random_state=0)
print(rfrModel(train_X,test_X,train_y,test_y))
    
    
