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

#------------------Standart Scale---------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(x)
Y = sc.fit_transform(y)


#-----------------PIPELINE SVR--------------------

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('svr', SVR(kernel='poly'))  
])

pipeline.fit(X, Y)



#----------------joblib------------------



joblib.dump(pipeline, "svr_pipeline.pkl")
print("The model has been saved")






