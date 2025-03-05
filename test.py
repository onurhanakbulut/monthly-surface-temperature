import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#----------------JOBLIB LOAD--------------------
lr_model = joblib.load('lr_model.pkl')
pr_pipeline = joblib.load('poly_pipeline.pkl')
svr_pipeline = joblib.load('svr_model.pkl')
dt_model = joblib.load('dt_model.pkl')
rf_model = joblib.load('rf_model.pkl')


#-------------------JOBLIB TEST LOAD------------------------
x_test = joblib.load("x_test.pkl")
y_test = joblib.load("y_test.pkl")



#------------------LINEAR REGRESSION EVALUATE------------------------
predict_lr = lr_model.predict(x_test)




r2 = r2_score(y_test, predict_lr)
mae = mean_absolute_error(y_test, predict_lr)
mse = mean_squared_error(y_test, predict_lr)
rmse = np.sqrt(mse)


print("📊 Linear Regression Model Performansı:")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


#---------------------POLYNOMIAL REGRESSON EVALUATE-----------------
predict_pr = pr_pipeline.predict(x_test)



r2 = r2_score(y_test, predict_pr)
mae = mean_absolute_error(y_test, predict_pr)
mse = mean_squared_error(y_test, predict_pr)
rmse = np.sqrt(mse)


print("📊 Polynomial Regression Model Performansı:")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


#------------------SVR EVALUEATE---------------------


predict_svr = svr_pipeline.predict(x_test)




r2 = r2_score(y_test, predict_svr)
mae = mean_absolute_error(y_test, predict_svr)
mse = mean_squared_error(y_test, predict_svr)
rmse = np.sqrt(mse)


print("📊 SVR Model Performansı:")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

#--------------------DECISION TREE EVALUEATE-----------------------

predict_dt = dt_model.predict(x_test)

r2 = r2_score(y_test, predict_dt)
mae = mean_absolute_error(y_test, predict_dt)
mse = mean_squared_error(y_test, predict_dt)
rmse = np.sqrt(mse)


print("📊 DT Model Performansı:")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

#------------------------RANDOM FOREST EVALUATE-----------------------
predict_rf = rf_model.predict(x_test)


r2 = r2_score(y_test, predict_rf)
mae = mean_absolute_error(y_test, predict_rf)
mse = mean_squared_error(y_test, predict_rf)
rmse = np.sqrt(mse)


print("📊 RF Model Performansı:")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")












