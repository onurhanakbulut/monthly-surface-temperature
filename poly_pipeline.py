import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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


#------------POLY------------------
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# pr = PolynomialFeatures(degree=5)
# poly=pr.fit_transform(x)
# lr_poly = LinearRegression()
# lr_poly.fit(poly,y)

# print('Poly Regression Turkey 2025 Summer -> ',lr_poly.predict(pr.fit_transform([[11.0812,2025,7]])))

#-----------PIPELINE-------------------------
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=5)),  
    ('linear_regression', LinearRegression())  
])




#----------------joblib------------------



joblib.dump(pipeline, "poly_pipeline.pkl")
print("The model has been saved")






