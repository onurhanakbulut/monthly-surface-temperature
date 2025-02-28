import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

#-------------PREPROCESS-------------------
data = pd.read_csv('average-monthly-surface-temperature.csv')
df = data

df['Day'] = pd.to_datetime(df['Day'], format ="%Y-%m-%d")
df['Day'] = df['Day'].dt.month
df = df.rename(columns={'Day':'Month'})

df = df.drop(['Code','Average surface temperature.1'], axis=1)


#-----------------TARGET ENCODING--------------------

df['Entity_Encoded'] = df.groupby('Entity')['Average surface temperature'].transform('mean')
df.insert(1,'Entity_Encoded',df.pop('Entity_Encoded'))


#-------------------
x = df.iloc[:,1:4].values
y = df.iloc[:,-1:].values

#--------------LINEAR REGRESSION-------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

#----------------joblib------------------



joblib.dump(lr, "lr_model.pkl")
print("The model has been saved")






