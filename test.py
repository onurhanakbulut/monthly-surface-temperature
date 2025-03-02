import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib



df = pd.read_csv('futuredata1.csv')
test = df.iloc[:,1:].values


lr_model = joblib.load('lr_model.pkl')
pr_pipeline = joblib.load('poly_pipeline.pkl')
svr_pipeline = joblib.load('svr_model.pkl')
dt_model = joblib.load('dt_model.pkl')
rf_model = joblib.load('rf_model.pkl')




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_test = sc.fit_transform(test)



predict_lr = lr_model.predict(test)
predict_pr = pr_pipeline.predict(test)
predict_svr = svr_pipeline.predict(scaled_test)
predict_dt = dt_model.predict(test)
predict_rf = rf_model.predict(test)




