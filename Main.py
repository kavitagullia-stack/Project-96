import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
data=pd.read_csv("Churn_Modelling.csv")
print(data.head())
print(data.info())
print(data.columns)
print(data.dtypes)
l=LabelEncoder()
data["Geography"]=l.fit_transform(data["Geography"])
data["Gender"]=l.fit_transform(data["Gender"])
print(data)
print(data.info())
data=data.drop(["RowNumber","CustomerId","Surname"],axis=1)
print(data.shape)
y=data.pop("Exited")
x=data
print(x.shape)
print(y.shape)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
s=StandardScaler()
xtrain=s.fit_transform(xtrain)
xtest=s.fit_transform(xtest)
ytrain=s.fit_transform(ytrain)
ytest=s.fit_transform(ytest)

