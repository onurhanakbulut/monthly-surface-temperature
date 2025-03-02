import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib





svr_pipeline = joblib.load('svr_model.pkl')
lr_pipeline = joblib.load('lr_model.pkl')
pr_pipeline = joblib.load('poly_pipeline.pkl')

x= np.array([[11.0812,2025,7]])
a=np.array([[11.0812,2025,7]])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a = sc.fit_transform(a)


print(lr_pipeline.predict([[11.0812,2025,7]]))
print(pr_pipeline.predict(x))
print(svr_pipeline.predict(a))













