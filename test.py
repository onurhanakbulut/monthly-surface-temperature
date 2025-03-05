import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv('futuredata1.csv')
mytest = df.iloc[:,1:].values


lr_model = joblib.load('lr_model.pkl')
pr_pipeline = joblib.load('poly_pipeline.pkl')
svr_pipeline = joblib.load('svr_model.pkl')
dt_model = joblib.load('dt_model.pkl')
rf_model = joblib.load('rf_model.pkl')



x_test = joblib.load("x_test.pkl")
y_test = joblib.load("y_test.pkl")

predict_lr = lr_model.predict(x_test)



# Performans değerlendirme
r2 = r2_score(y_test, predict_lr)
mae = mean_absolute_error(y_test, predict_lr)
mse = mean_squared_error(y_test, predict_lr)
rmse = np.sqrt(mse)

# Sonuçları yazdır
print("📊 Linear Regression Model Performansı:")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")













# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# scaled_test = sc.fit_transform(test)





# predict_lr = lr_model.predict(test)
# predict_pr = pr_pipeline.predict(test)
# predict_svr = svr_pipeline.predict(scaled_test)
# predict_dt = dt_model.predict(test)
# predict_rf = rf_model.predict(test)




