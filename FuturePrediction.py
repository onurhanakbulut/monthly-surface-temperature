import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#-------------MY TEST DATA----------------
df = pd.read_csv('futuredata1.csv')
mytest = df.iloc[:,1:].values

#----------------JOBLIB LOAD--------------------
pr_pipeline = joblib.load('poly_pipeline.pkl')




#----------------POLYNOMIAL PREDICTION (0.8423)-----------
predict_pr = pr_pipeline.predict(mytest)





#------------------CSV---------------------------------
predict_pr = pd.DataFrame(predict_pr)

prediction = pd.concat([df,predict_pr], ignore_index=True, axis=1)

prediction = prediction.drop(columns=[1])


prediction.columns = ['Country', 'Year', 'Month', 'Average Surface Temperature']

prediction.to_csv('FuturePrediction.csv', index=False)




#------------------------analyze visualization------------------
from plotly import express

express.scatter(prediction, x='Month', y='Average Surface Temperature', trendline='lowess').show(renderer='iframe_connected')






# #------------------------analyze visualization------------------
# from plotly import express

# express.scatter(data_frame=df[df['Entity'] == 'World'].drop(columns=['Code', 'Day', 'Average surface temperature']).drop_duplicates()
# , x='year', y='Average surface temperature.1', trendline='lowess').show(renderer='iframe_connected',)








