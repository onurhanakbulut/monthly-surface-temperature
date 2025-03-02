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


#---------------------------RANDOMIZEDSEARCHCV-------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()

# # Hiperparametreler
# param_dist = {
#     'n_estimators': [50, 100, 200, 300, 500],
#     'max_depth': [5, 10, 20, 30, None],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 5, 10, 20]
# }

# # 🚀 RandomizedSearchCV ile en iyi parametreleri bul
# random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
#                                     n_iter=10, cv=3, scoring='neg_mean_squared_error', 
#                                     verbose=1, n_jobs=-1, random_state=42)

# random_search.fit(x,y)

# # En iyi hiperparametreleri göster
# print(f"En iyi parametreler: {random_search.best_params_}")

# # En iyi modeli seç
# best_model = random_search.best_estimator_
# print(best_model)

#En iyi parametreler: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 5}
# RandomForestRegressor(max_depth=5, min_samples_leaf=5, min_samples_split=10,
#                       n_estimators=300)



#-------------------RANDOM FOREST------------------

rf = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=5 ,n_jobs=-1)
rf.fit(x,y)



# #-----------------JOBLIB-----------------------
joblib.dump(rf, "rf_model.pkl")
print("The model has been saved")





