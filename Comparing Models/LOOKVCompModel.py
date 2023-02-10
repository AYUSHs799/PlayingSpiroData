from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
import pandas as pd
import numpy as np

# Loading npy data
X = pd.DataFrame(np.load("PlayingSpiroData/Spiro-Data/npy_file/FEV1_FEATURES_60.npy"))
y = pd.DataFrame(np.load("PlayingSpiroData/Spiro-Data/npy_file/FEV1_LABELS_60.npy"))

# 
RF = RandomForestRegressor( n_jobs=-1, bootstrap=True, criterion='absolute_error', 
                                  n_estimators=100,  max_features=1.0, max_depth=300,  
                                  min_samples_leaf=1, min_samples_split=5 ,random_state=42)

DT = DecisionTreeRegressor( criterion='absolute_error',min_samples_leaf=1, 
                                min_samples_split=5 ,random_state=42)

LR = LinearRegression()

RegDict = { "Random_Forrest_Regression" :   RF,
            "Decision_Tree_Regression" :    DT,
            "Linear_Regression":            LR }

abserror = []
tot = len(X)
loo = LeaveOneOut()


for Reg in RegDict:

    print("For : ",Reg)
    Model = RegDict[Reg]
    prog = 0
    y_GT = []
    y_PT = []

    for i,(train_index, test_index) in enumerate(loo.split(X)):
        prog = prog + 1
        print("Progress : {0}/{1}\r".format(prog,tot),end=" ")

        X_Train, X_Test = X.iloc[train_index],X.iloc[test_index]
        y_Train, y_Test = y.iloc[train_index],y.iloc[test_index]
        
        Model.fit(X_Train, np.ravel(y_Train))
        pred = Model.predict(X_Test)   
        y_GT.append(y_Test.iloc[0,0])
        y_PT.append(pred[0])  

        abserror.append(np.abs( (y_Test.iloc[0,0] - pred[0])/ y_Test.iloc[0,0] ))

    MAPE = 100 * np.mean(abserror)
    print("Calculated MAPE : ", MAPE )
    print(" sklearm MAPE : " , 100 * mean_absolute_percentage_error(y_GT,y_PT))





    