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


#--------------------TRAIN TEST------------------------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2, shuffle=False)


#----------------DECISION TREE------------------------
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=10)
dt.fit(x_train,y_train)



#------------------------------JOBLIB------------------------
joblib.dump(dt, "dt_model.pkl")
print("The model has been saved")







