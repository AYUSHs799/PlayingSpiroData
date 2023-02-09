import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

#Loading Data from npy files
X_all = pd.DataFrame(np.load("C:/Users/HP/OneDrive/Documents/NipunBatraResearch/Spiro-Data/npy_file/FVC_FEATURES_60.npy"))
y = pd.DataFrame(np.load("C:/Users/HP/OneDrive/Documents/NipunBatraResearch/Spiro-Data/npy_file/FVC_LABELS_60.npy"))

# X_used = X_all.iloc[:,:100]
X_used = X_all

#70:30 Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(X_used,y,test_size=0.7,random_state=42)

reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train,y_train)
y_hat = reg.predict(X_test)

print("Decision Tree regression for FVC")
print("RMSE obtained :",mean_squared_error(y_test,y_hat))
print("MAPE obtained :",mean_absolute_percentage_error(y_test,y_hat))
print("MEA obtained :",mean_absolute_error(y_test,y_hat))


